import time
import numpy as np
import torch
import os 
import onnxruntime as ort
from tokenizer import (
    Tokenizer,
    AUDIO_OFFSET,
    NUM_AUDIO_TOKENS,
    END_OF_SPEECH_TOKEN_ID,
    START_OF_SPEECH_TOKEN_ID,
    CODEC_FRAME_RATE
)
from text.text_normalizer import split_text_whitespace, normalize_text

class KaoPadTTS:
    def __init__(self, model_id: str = "VIZINTZOR/KaoPadTTS-85M", local_path: str = None, device: str = "cpu"):

        self.device = device
        providers = self._get_providers(device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        enc_name = "encoder.onnx"
        dec_name = "decoder.onnx"
        subfolder = "onnx"

        if local_path is None:
            from huggingface_hub import hf_hub_download
            enc_path = hf_hub_download(repo_id=model_id, filename=enc_name, subfolder=subfolder)
            dec_path = hf_hub_download(repo_id=model_id, filename=dec_name, subfolder=subfolder)
        else:
            enc_path = os.path.join(local_path, enc_name)
            dec_path = os.path.join(local_path, dec_name)

        # Load ONNX sessions
        self.enc_sess = ort.InferenceSession(enc_path, opts, providers=providers)
        self.dec_sess = ort.InferenceSession(dec_path, opts, providers=providers)

        n_out = len(self.dec_sess.get_outputs())
        self.n_layers = (n_out - 1) // 2

        print(f"  ✅ encoder.onnx")
        print(f"  ✅ decoder.onnx  ({self.n_layers} layers)")

    def _get_providers(self, device: str):
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        outs = self.enc_sess.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
        n = self.n_layers
        enc_out = outs[0]
        cross_ks = outs[1:1 + n]
        cross_vs = outs[1 + n:]
        return enc_out, cross_ks, cross_vs

    def decode(
        self,
        input_ids: np.ndarray,
        speaker_emb: np.ndarray,
        attention_mask: np.ndarray,
        cross_ks: list,
        cross_vs: list,
        self_ks: list,
        self_vs: list
    ):
        n = self.n_layers
        feed = {
            "input_ids": input_ids,
            "speaker_emb": speaker_emb,
            "attention_mask": attention_mask
        }
        for i in range(n):
            feed[f"cross_k_{i}"] = cross_ks[i]
            feed[f"cross_v_{i}"] = cross_vs[i]

        for i in range(n):
            if self_ks is not None and self_vs is not None:
                feed[f"self_k_{i}"] = self_ks[i]
                feed[f"self_v_{i}"] = self_vs[i]
            else:
                shape_k = cross_ks[i].shape[:2] + (1,) + cross_ks[i].shape[3:]
                shape_v = cross_vs[i].shape[:2] + (1,) + cross_vs[i].shape[3:]
                feed[f"self_k_{i}"] = np.zeros(shape_k, dtype=np.float32)
                feed[f"self_v_{i}"] = np.zeros(shape_v, dtype=np.float32)

        outs = self.dec_sess.run(None, feed)
        logits = outs[0]
        new_self_ks = outs[1:1 + n]
        new_self_vs = outs[1 + n:]
        return logits, new_self_ks, new_self_vs

    def _apply_audio_mask(self, logits: np.ndarray) -> np.ndarray:
        masked = np.full_like(logits, -np.inf)
        masked[:, AUDIO_OFFSET:AUDIO_OFFSET + NUM_AUDIO_TOKENS] = \
            logits[:, AUDIO_OFFSET:AUDIO_OFFSET + NUM_AUDIO_TOKENS]
        masked[:, END_OF_SPEECH_TOKEN_ID] = logits[:, END_OF_SPEECH_TOKEN_ID]
        return masked

    def _rep_penalty(self, logits: np.ndarray, recent_tokens: list, penalty: float) -> np.ndarray:
        logits = logits.copy()
        for tid in set(recent_tokens[-100:]):
            if AUDIO_OFFSET <= tid < AUDIO_OFFSET + NUM_AUDIO_TOKENS:
                logits[:, tid] /= penalty
        return logits

    def _top_k_filter(self, logits: np.ndarray, top_k: int) -> np.ndarray:
        if top_k <= 0:
            return logits
        logits = logits.copy()
        k = min(top_k, logits.shape[-1])
        kth_vals = np.partition(logits, -k, axis=-1)[:, -k]
        logits[logits < kth_vals[:, None]] = -np.inf
        return logits

    def _top_p_filter(self, logits: np.ndarray, top_p: float) -> np.ndarray:
        if top_p >= 1.0:
            return logits
        logits = logits.copy()
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        sorted_idx = np.argsort(-probs, axis=-1)
        sorted_p = np.take_along_axis(probs, sorted_idx, axis=-1)
        cum_p = np.cumsum(sorted_p, axis=-1)
        remove = cum_p > top_p
        remove[:, 1:] = remove[:, :-1].copy()
        remove[:, 0] = False
        remove_orig = np.zeros_like(remove)
        np.put_along_axis(remove_orig, sorted_idx, remove, axis=-1)
        logits[remove_orig] = -np.inf
        return logits

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def _sample(self, probs: np.ndarray) -> int:
        return int(np.random.choice(probs.shape[-1], p=probs[0]))

    def _process_logits(self, logits_step, generated_tokens, temperature, top_k, top_p, rep_penalty):
        lg = logits_step[:, 0, :]
        lg = self._apply_audio_mask(lg)
        if rep_penalty != 1.0 and generated_tokens:
            lg = self._rep_penalty(lg, generated_tokens, rep_penalty)
        lg = lg / temperature
        lg = self._top_k_filter(lg, top_k)
        lg = self._top_p_filter(lg, top_p)
        return self._sample(self._softmax(lg))

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker_emb,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 250,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
    ) -> torch.Tensor | None:

        tokenizer = Tokenizer()

        enc_ids = tokenizer.build_encoder_input(text).unsqueeze(0)
        enc_ids_np = enc_ids.numpy().astype(np.int64)
        enc_mask_np = np.ones_like(enc_ids_np, dtype=np.int64)
        _, cross_ks, cross_vs = self.encode(enc_ids_np, enc_mask_np)

        if isinstance(speaker_emb, torch.Tensor):
            spk_np = speaker_emb.unsqueeze(0).float().numpy()
        else:
            spk_np = np.asarray(speaker_emb, dtype=np.float32)[None]

        generated_tokens = []
        cur_self_ks = None
        cur_self_vs = None

        bos_ids = np.array([[START_OF_SPEECH_TOKEN_ID]], dtype=np.int64)
        logits, cur_self_ks, cur_self_vs = self.decode(
            bos_ids, spk_np, enc_mask_np, cross_ks, cross_vs, cur_self_ks, cur_self_vs
        )
        tok_id = self._process_logits(logits, generated_tokens, temperature, top_k, top_p, rep_penalty)
        if tok_id == END_OF_SPEECH_TOKEN_ID:
            return None
        generated_tokens.append(tok_id)

        t0 = time.time()
        try:
            from tqdm import tqdm
            iterator = tqdm(range(max_new_tokens - 1), desc="Generating tokens")
        except ImportError:
            iterator = range(max_new_tokens - 1)

        for _ in iterator:
            cur_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
            logits, cur_self_ks, cur_self_vs = self.decode(
                cur_ids, spk_np, enc_mask_np, cross_ks, cross_vs, cur_self_ks, cur_self_vs
            )
            tok_id = self._process_logits(logits, generated_tokens, temperature, top_k, top_p, rep_penalty)
            if tok_id == END_OF_SPEECH_TOKEN_ID:
                break
            generated_tokens.append(tok_id)

        elapsed = time.time() - t0
        if generated_tokens:
            dur_sec = len(generated_tokens) / CODEC_FRAME_RATE
            print(f"Generated {len(generated_tokens)} tokens "
                  f"({dur_sec:.2f}s audio) in {elapsed:.2f}s | "
                  f"RTF {elapsed / max(dur_sec, 1e-6):.3f}")
        else:
            print("No tokens generated.")
            return None

        result = torch.tensor(generated_tokens, dtype=torch.long)
        audio_mask = (result >= AUDIO_OFFSET) & (result < AUDIO_OFFSET + NUM_AUDIO_TOKENS)
        return result[audio_mask] - AUDIO_OFFSET
    
    @torch.inference_mode()
    def generate_batch(self, text, speaker_emb, 
        temperature=0.7, top_k=250, top_p=0.95, rep_penalty=1.1, max_new_tokens=512, max_chars=150
    ):
        clean_text = normalize_text(text)
        chunks = split_text_whitespace(clean_text, max_chars=max_chars)
        for idx, i in enumerate(chunks):
            print(idx, ":", i)
        outputs = []

        for chunk in chunks:
            audio_tokens = self.generate(
                text=chunk,
                speaker_emb=speaker_emb,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rep_penalty=rep_penalty,
            )
            if audio_tokens is not None:
                outputs.append(audio_tokens)

        if not outputs:
            return None

        crossfade = 1
        if len(outputs) == 1:
            merged_codes = outputs[0]
        elif isinstance(outputs[0], torch.Tensor):
            merged_codes = outputs[0][:-crossfade].clone()
            for codes in outputs[1:]:
                merged_codes = torch.cat((merged_codes, codes[crossfade:]), dim=-1)
        else:
            merged_codes = outputs[0][:-crossfade]
            for codes in outputs[1:]:
                merged_codes.extend(codes[crossfade:])

        return merged_codes
import torch
import soundfile as sf
from pathlib import Path
from typing import Optional

class CODEC:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.sample_rate = 44_100 
        self.codebook_size = 12_800 
        self.frame_rate = 25.0 
        self._load_model()

    def _load_model(self):
        from miocodec import MioCodecModel
        self.model = MioCodecModel.from_pretrained("Aratako/MioCodec-25Hz-44.1kHz-v2")
        self.model = self.model.to(self.device).eval()
        print("Loaded Codec Model!")

    @torch.no_grad()
    def encode(self, wav_path: str | Path) -> dict:
        """
        Encode wav file → MioCodec codes + global_embedding.
        """
        data, sr = sf.read(str(wav_path), dtype='float32')
        waveform = torch.from_numpy(data)
        return self.encode_waveform(waveform, sr)

    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor, sr: int) -> dict:
        """
        Encode directly from waveform tensor.
        waveform: [samples] or [channels, samples]
        sr: int
        """
        if waveform.dim() == 2:  # stereo
            waveform = waveform.mean(1)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, samples]
            
        if sr != self.sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        audio = waveform.to(self.device).float()

        # MioCodec encode returns (content_token_indices, global_embedding)
        result = self.model.encode(audio)
        codes = result.content_token_indices.squeeze().cpu()       # [num_frames]
        global_emb = result.global_embedding.squeeze().cpu()       # [128]

        return {
            'codes': codes,
            'global_embedding': global_emb,
        }

    @torch.no_grad()
    def decode(self, codes: torch.Tensor,
               global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode MioCodec codes → waveform.

        Args:
            codes: [num_frames] — token indices in [0, 12799]
            global_embedding: [128] — speaker embedding

        Returns:
            waveform: [samples] float32
        """
        codes = codes.to(self.device)
        global_embedding = global_embedding.to(self.device)

        if codes.dim() > 1:
            codes = codes.squeeze()
        if global_embedding.dim() > 1:
            global_embedding = global_embedding.squeeze()

        audio = self.model.decode(
            global_embedding=global_embedding,
            content_token_indices=codes,
        )
        return audio.squeeze().cpu().float()

    def encode_to_tokens(self, wav_path: str) -> dict:
        """Convenience: encode and return codes + embedding."""
        return self.encode(wav_path)

    def tokens_to_wav(self, codes: torch.Tensor,
                      global_embedding: torch.Tensor,
                      output: Optional[str] = None) -> torch.Tensor:
        """Decode tokens to wav, optionally save."""
        wav = self.decode(codes, global_embedding)
        if output:
            sf.write(output, wav.numpy(), self.sample_rate)
        return wav
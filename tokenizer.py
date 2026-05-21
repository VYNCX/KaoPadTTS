import re
import torch

TH_CHARS = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไๅํ็่้๊๋ฯฺๆ์ํ๎๏๚๛๐๑๒๓๔๕๖๗๘๙฿'
EN_LOWER  = "abcdefghijklmnopqrstuvwxyz"
EN_UPPER  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS    = "0123456789"
PUNCT     = '.,!?;:-–—…"\'()[]{}«»„"" '
EXTRA     = "\n\t"

_ALL_CHARS: list[str] = []
_seen: set[str] = set()
for _src in [TH_CHARS, EN_LOWER, EN_UPPER, DIGITS, PUNCT, EXTRA]:
    for _ch in _src:
        if _ch not in _seen:
            _ALL_CHARS.append(_ch)
            _seen.add(_ch)

SPECIAL_TOKENS = {
    "<pad>":             0,
    "<start_of_text>":   1,
    "<end_of_text>":     2,
    "<start_of_speech>": 3,
    "<end_of_speech>":   4,
    "<spk_0>":           5,  # kept for compatibility, but speaker embedding is primary
    "<spk_1>":           6,
    "<spk_2>":           7,
    "<spk_3>":           8,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)     # 9

# ── Vocab offsets ───────────────────────────────────────────────
TEXT_CHARS       = _ALL_CHARS
TEXT_VOCAB_SIZE  = len(TEXT_CHARS)             # ~146
TEXT_OFFSET      = NUM_SPECIAL_TOKENS         # 9
AUDIO_OFFSET     = TEXT_OFFSET + TEXT_VOCAB_SIZE  # 155
CODEC_CODEBOOK_SIZE = 12800
NUM_AUDIO_TOKENS = CODEC_CODEBOOK_SIZE            # 12,800
TOTAL_VOCAB_SIZE = AUDIO_OFFSET + NUM_AUDIO_TOKENS  # 12,955

# Encoder needs only text vocab; decoder needs full vocab
ENCODER_VOCAB_SIZE = AUDIO_OFFSET      # 155 (special + text)
DECODER_VOCAB_SIZE = TOTAL_VOCAB_SIZE  # 12,955 (full)

# ── Convenience IDs ─────────────────────────────────────────────
PAD_TOKEN_ID             = SPECIAL_TOKENS["<pad>"]
START_OF_TEXT_TOKEN_ID   = SPECIAL_TOKENS["<start_of_text>"]
END_OF_TEXT_TOKEN_ID     = SPECIAL_TOKENS["<end_of_text>"]
START_OF_SPEECH_TOKEN_ID = SPECIAL_TOKENS["<start_of_speech>"]
END_OF_SPEECH_TOKEN_ID   = SPECIAL_TOKENS["<end_of_speech>"]
SPK_0_TOKEN_ID           = SPECIAL_TOKENS["<spk_0>"]
SPK_1_TOKEN_ID           = SPECIAL_TOKENS["<spk_1>"]

# ── Helper functions ────────────────────────────────────────────
def audio_token_id(code: int) -> int:
    """MioCodec code → global token ID."""
    return AUDIO_OFFSET + code

def decode_audio_token(token_id: int) -> int:
    """Global token ID → MioCodec code."""
    return token_id - AUDIO_OFFSET

def is_audio_token(token_id: int) -> bool:
    return AUDIO_OFFSET <= token_id < AUDIO_OFFSET + NUM_AUDIO_TOKENS

def is_special_token(token_id: int) -> bool:
    return 0 <= token_id < NUM_SPECIAL_TOKENS

def is_text_token(token_id: int) -> bool:
    return TEXT_OFFSET <= token_id < AUDIO_OFFSET

class Tokenizer:
    def __init__(self):
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        for i, ch in enumerate(TEXT_CHARS):
            tid = TEXT_OFFSET + i
            self.char2id[ch] = tid
            self.id2char[tid] = ch

        self._special_id_to_name = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.vocab_size = TOTAL_VOCAB_SIZE
        self.text_vocab_size = len(TEXT_CHARS)

    def normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'[«»„""]', '"', text)
        return text

    def encode_text(self, text: str) -> list[int]:
        text = self.normalize_text(text)
        return [self.char2id[ch] for ch in text if ch in self.char2id]

    def decode_text(self, ids: list[int]) -> str:
        return "".join(self.id2char.get(t, "") for t in ids if is_text_token(t))

    def build_encoder_input(self, text: str) -> torch.Tensor:
        """
        Encoder input: <sot> text_chars <eot>
        No speaker token — speaker info comes from embedding.
        """
        text_ids = self.encode_text(text)
        seq = text_ids
        return torch.tensor(seq, dtype=torch.long)
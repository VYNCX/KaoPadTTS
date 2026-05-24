# KaoPadTTS
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/VIZINTZOR/KaoPadTTS-85M)

KaoPadTTS: Text-to-Speech ภาษาไทยแบบ Autoregressive ขนาด 85M ที่รองรับ Voice Cloning เพื่อสร้างเสียงพูดที่เป็นธรรมชาติ ขนาดเล็ก สามารถใช้งานได้บน CPU,GPU

## โมเดล

| Model Name | Parameters | Codec |
|---|---|---|
| [KaoPadTTS-85M](https://huggingface.co/VIZINTZOR/KaoPadTTS-85M) | 85 M | [MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) 

## การติดตั้ง 

```bash
pip install git+https://github.com/VYNCX/KaoPadTTS.git
```

## ใช้งาน

```python
from KaoPadTTS import KaoPadTTS
from KaoPadTTS.codec import CODEC

device = "cpu"

tts = KaoPadTTS(device=device)
codec = CODEC(device=device)
ref_audio = "sample.wav"
ref = codec.encode(ref_audio)
speaker_emb = ref["global_embedding"].to(device)
text = """สวัสดีครับ นี่คือเสียงพูดภาษาไทย"""
audio_tokens = tts.generate_batch(text,
                                  speaker_emb=speaker_emb,
                                  temperature=0.3,
                                  top_k=150,
                                  top_p=0.95,
                                  rep_penalty=1.1,
                                  max_new_tokens=300)
output_wav = "output.wav"
codec.tokens_to_wav(audio_tokens, speaker_emb, output_wav)
```

#!/usr/bin/env python
"""open-music-llm: MusicGen(Meta AudioCraft、EnCodecトークン+Transformer次トークン予測の
実在するオーディオLLM)をコマンドラインから呼べるようにした薄いラッパー。

正直な開示: モデル重み(facebook/musicgen-*)はCC-BY-NC 4.0(非商用限定)。
"""
import argparse
import sys

import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration


def main() -> int:
    p = argparse.ArgumentParser(description="MusicGenでテキストから音楽・効果音を生成する")
    p.add_argument("prompt", help="生成したい内容の説明(英語推奨)")
    p.add_argument("output", help="出力先WAV")
    p.add_argument("--model", default="facebook/musicgen-small", help="Hugging Faceのモデル名")
    p.add_argument("--max-new-tokens", type=int, default=256, help="256トークン≈5秒相当")
    args = p.parse_args()

    print(f"loading {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model)
    print("model loaded, generating...", flush=True)

    inputs = processor(text=[args.prompt], padding=True, return_tensors="pt")
    audio_values = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    sampling_rate = model.config.audio_encoder.sampling_rate

    audio = audio_values[0, 0].numpy()
    scipy.io.wavfile.write(args.output, rate=sampling_rate, data=audio)
    print(f"wrote {args.output} ({sampling_rate}Hz, {len(audio)/sampling_rate:.2f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""open-music-llm: MusicGen(Meta AudioCraft、EnCodecトークン+Transformer次トークン予測の
実在するオーディオLLM)をコマンドラインから呼べるようにした薄いラッパー。

正直な開示: モデル重み(facebook/musicgen-*)はCC-BY-NC 4.0(非商用限定)。
このスクリプトは商用利用を検出する手段を持たない(利用目的の技術的な検証は不可能)ため、
「非商用利用であることの明示的な確認」を毎回要求する形で警告機構を実装している。
"""
import argparse
import io
import os
import sys

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

NON_COMMERCIAL_ENV = "OPEN_MUSIC_LLM_NON_COMMERCIAL_USE"

LICENSE_NOTICE = """\
============================================================
open-music-llm: MusicGenの学習済み重み(facebook/musicgen-*)は
CC-BY-NC 4.0(非商用限定)ライセンスです。商用利用はできません。
商用化を検討する場合は、自前でのファインチューニング/学習
(狭い領域限定ならコンシューマGPU数枚規模で現実的、詳細はCLAUDE.md参照)
または商用利用可能な別モデルへの切り替えが必要です。
============================================================\
"""


def check_non_commercial_confirmation(confirmed_flag: bool) -> None:
    """非商用利用の明示的な確認が無ければ生成を拒否する(商用利用検出は技術的に不可能なため)。"""
    print(LICENSE_NOTICE, file=sys.stderr, flush=True)
    if confirmed_flag or os.environ.get(NON_COMMERCIAL_ENV) == "1":
        return
    print(
        "\nエラー: 非商用利用であることの確認が必要です。\n"
        "  --confirm-non-commercial-use を付けて実行するか、\n"
        f"  環境変数 {NON_COMMERCIAL_ENV}=1 を設定してから実行してください。\n"
        "商用利用の場合は、上記ライセンス通知に従い自前学習/別モデルへの切り替えを検討してください。",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser(description="MusicGenでテキストから音楽・効果音を生成する")
    p.add_argument("prompt", help="生成したい内容の説明(英語推奨)")
    p.add_argument("output", help="出力先WAV")
    p.add_argument("--model", default="facebook/musicgen-small", help="Hugging Faceのモデル名")
    p.add_argument("--max-new-tokens", type=int, default=256, help="256トークン≈5秒相当")
    p.add_argument(
        "--confirm-non-commercial-use",
        action="store_true",
        help=f"非商用利用であることを確認する(または環境変数{NON_COMMERCIAL_ENV}=1)",
    )
    args = p.parse_args()

    check_non_commercial_confirmation(args.confirm_non_commercial_use)

    # 重いimportは確認チェックの後に行う(未確認時に torch/transformers の読み込みで待たされないため)
    import scipy.io.wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

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

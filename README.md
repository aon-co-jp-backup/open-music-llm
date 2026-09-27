# open-music-llm

**MusicGen**([facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)、
コードMIT・学習済み重み`facebook/musicgen-*`は**CC-BY-NC 4.0(非商用限定)**)という実在する
「オーディオLLM」(EnCodecで音声を離散トークン化し、通常のLLMと同じTransformerデコーダ+
次トークン予測で音楽・効果音を生成する)を、Rust + RPoemでラップする。

## これは何か

- 帯域拡張(音声超解像)は[`open-audio-sr`](https://github.com/aon-co-jp/open-audio-sr)
  (拡散モデルのAudioSR)が担当。こちらは**テキストから新しい音楽・効果音を生成する**用途。
- 詳しい経緯・調査結果は[CLAUDE.md](CLAUDE.md)参照(「LLMは音声を生成できない」という
  以前の説明の訂正・自前学習の現実的な規模感など)。

## ライセンスについて(重要)

`facebook/musicgen-*`の学習済み重みは**CC-BY-NC 4.0(非商用限定)**。現状は非商用利用のみ
を前提に開発している。商用化を検討する段階になったら、自前学習(狭い領域限定ならコンシューマ
GPU4〜16枚規模で現実的、詳細はCLAUDE.md)またはライセンス許諾済みの別モデルへの切り替えが
必要になる——**この警告を実際に出す仕組みはまだ未実装**(次回の課題)。

## 現状(2026-09-27)

**実際の音楽生成を確認済み。** `facebook/musicgen-small`で"gentle lo-fi piano melody"から
5.06秒・32kHzのWAVを生成し、`ffmpeg volumedetect`で無音でない実音声(mean -23.1dB /
max -6.1dB)であることを確認した。環境固有の問題(`transformers`のimportが遅い件)は
`USE_TF=0 USE_FLAX=0`環境変数指定で回避できた(詳細は[CLAUDE.md](CLAUDE.md)参照)。

## 使い方

```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install "transformers>=4.31.0" scipy

$env:USE_TF="0"; $env:USE_FLAX="0"
.venv\Scripts\python.exe python\generate.py "gentle lo-fi piano melody" out.wav
```

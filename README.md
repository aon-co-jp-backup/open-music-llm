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

## 現状(2026-09-26)

**Python環境構築まで完了、実際の音楽生成は未検証のまま中断。** 次回は`python/generate.py`
の実行から再開すること(環境固有の問題は[CLAUDE.md](CLAUDE.md)参照)。

## 使い方(予定、生成自体は未検証)

```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install "transformers>=4.31.0" scipy

.venv\Scripts\python.exe python\generate.py "gentle lo-fi piano melody" out.wav
```

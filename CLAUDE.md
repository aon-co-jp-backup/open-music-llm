# 開発方針(open-music-llm)

MusicGen(Meta AudioCraft、EnCodecトークン+Transformer次トークン予測の実在するオーディオLLM)を
Rust+RPoemでラップし、テキストから音楽・効果音を生成する。コードはMIT、**学習済み重みは
CC-BY-NC 4.0(非商用限定)**。

## なぜLLMアーキテクチャで音声を生成できるのか(2026-09-26調査、正直な訂正)

以前「LLM(言語モデル)は音声波形を生成できない」と説明したが不正確だった。正確には:
**「テキストの次単語予測を行う通常のLLM」は音声を直接生成できないが、「EnCodec等の
ニューラル音声コーデックで音声を離散トークン列に変換し、そのトークン列に対して通常の
LLMと全く同じTransformerデコーダ+次トークン予測を適用する」というオーディオLLMの設計は
実在し、既に主流の一つ**(MusicGen/AudioGenがこの設計)。`aruaru-llm`基盤の
`open-cuda-llm`(GPT-2/Qwen2/DeepSeek: RoPE・GQA・RMSNorm・SwiGLU)はアーキテクチャ的には
このバックボーンに転用可能。

ただし**帯域拡張(音声超解像)という用途に限っては、2026年の研究で「オーディオLLM(離散
トークン+次トークン予測)は拡散モデルに性能面で劣る」と明記されている**
([A Survey of Advancing Audio Super-Resolution...](https://arxiv.org/html/2605.16681v1))。
そのため帯域拡張は[`open-audio-sr`](https://github.com/aon-co-jp/open-audio-sr)(拡散モデル
のAudioSR)、音楽・効果音の生成はこちら(オーディオLLMのMusicGen)、と役割を分けている。

## 自前学習(非商用ライセンス卒業)の現実的な規模感(2026-09-26調査)

- MusicGen本家の正確なGPU台数・時間はMeta非公開。類似規模の音声生成モデル(長尺音楽生成の
  拡散モデル)の実例では「CLAP事前学習: 約3,000 GPU時間・オートエンコーダ: 約16,000 GPU時間・
  拡散モデル本体: 約70,000 GPU時間」という報告あり
  ([Long-form music generation with latent diffusion](https://arxiv.org/pdf/2404.10301))。
  **つまり「数万枚のGPU」ではなく「数十〜数百枚のA100/H100級を数日〜数週間」規模**
  (GPT-4/Llama3クラスの数万枚規模とは1桁以上違う)。
- 現実的な道筋: ①MusicGenと同等品質を一から学習は数十〜数百枚規模で非現実的、②**既存公開
  モデルのファインチューニング(LoRA等)はコンシューマGPU数枚で現実的**
  ([ylacombe/musicgen-dreamboothing](https://github.com/ylacombe/musicgen-dreamboothing)が実例)、
  ③**狭い領域(自社データ・特定ジャンル)限定の小さい独自モデルを一から学習するのは
  コンシューマGPU4〜16枚・数週間規模で現実的**。
- 方針(ユーザー決定): 今は非商用(CC-BY-NC 4.0のMusicGen重み)で開始し、商用化が本格的に
  視野に入った段階で警告を出す仕組みを用意する(**未実装**、次回着手)。

## HANDOFF(2026-09-27 実際の生成に成功)

- リポジトリ新設・Python venv構築(64bit Python 3.13)・`torch`(CPU版)インストール済み。
- `transformers`のインストール自体は完了(`transformers==5.17.0`)。
- **実際の音楽生成を確認済み**: `USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1`を
  付けて`python/generate.py "gentle lo-fi piano melody" testdata/gen4.wav`を実行し、
  5.06秒・32kHzのWAVを生成。`ffmpeg -af volumedetect`で`mean_volume: -23.1dB`・
  `max_volume: -6.1dB`(無音ではない実際の音声)であることを確認した。
  モデルは`facebook/musicgen-small`(初回実行時にHugging Face Hubから自動ダウンロード)。
- **前回ハマった「`transformers`のimportが90秒で終わらない」問題の実際の原因**:
  importの遅さそのものは変わらず(数分規模)残っていたが、真の落とし穴は別にあった——
  `run_in_background`のシェルコマンド内でさらに`&`を付けて二重にバックグラウンド化すると、
  ラッパーシェル自身が起動直後(PID表示のみ)に`exit code 0`で終了してしまい、実際の
  Python処理の完了を待たずに「完了」と誤通知される。**対策: バックグラウンド実行したい
  コマンドは、ツール自体のバックグラウンド機能に任せ、コマンド文字列側に`&`を追加しない
  (ただの`python ... > log 2>&1`のみをバックグラウンド実行する)。**
- **正直な開示**: 「非商用限定でよいから警告機構を後で」という設計のうち、警告機構
  そのものはまだ1行も実装していない(次回の課題、下記参照)。

## 環境の既知の問題(重要、次回の作業効率のため)

このセッションで`torch`単体のimportに約30〜40秒かかることを確認済み(このマシン固有の
特性、原因未特定。ディスク連続読み書きは736MB/sと正常、Windows Defenderのリアルタイム
保護は無効化済みと確認済み)。**`from transformers import AutoProcessor`は90秒のタイムアウト
内に完了しなかった**(`torch`単体より明らかに重い)。次回はこの`transformers`のimportが
実際に完了するまで、より長いタイムアウト(数分〜十数分)で待つか、`transformers`の
バックエンド自動検出を抑制する環境変数(`TRANSFORMERS_NO_ADVISORY_WARNINGS=1`・
`USE_TF=0`・`USE_FLAX=0`等、未検証)を試すこと。詳細は[`open-audio-sr`のCLAUDE.md]
(https://github.com/aon-co-jp/open-audio-sr/blob/main/CLAUDE.md)にも同種の環境メモあり。

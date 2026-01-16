# Hist exog特徴量設計

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 12/15/2025 8:14:33  
**Updated:** 12/15/2025 8:24:09  
**Exported:** 12/15/2025 8:24:27  
**Link:** [https://chatgpt.com/c/693f44d9-1cd4-8323-bbf1-8d6a70a9570b](https://chatgpt.com/c/693f44d9-1cd4-8323-bbf1-8d6a70a9570b)  

## Prompt:
# hist_exog_list 要件定義書（NeuralForecast向け：過去外生変数）

## 1. 目的

* public.loto_hist に保存されている時系列（loto, unique_id, ds, y）から、NeuralForecast の hist_exog_list として利用可能な「過去由来の外生変数（hist exogenous）」を生成し、学習・推論に再利用できる形で永続化する。
* 生成された hist exog は、将来データを参照しない（リークしない）ことを第一要件とする。

## 2. スコープ

### 2.1 対象

* 入力テーブル：public.loto_hist
* 出力テーブル：public.loto_hist_exog（新規作成または再生成）
* 対象キー：(loto, unique_id, ds) 単位の時系列

### 2.2 非対象

* futr_exog_list（未来外生変数）の作成（既に public.loto_futr で実施済み）
* モデル学習・予測の実行そのもの

## 3. 用語

* **hist_exog**：予測時点より過去の情報から計算できる外生変数（例：ラグ、移動平均、異常スコア）。
* **リーク（データリーク）**：時点 t の特徴量計算に t 以降の y を混入させること。
* **窓（window）**：ローリング統計・クラスタリングなどで用いる過去区間（例：直近 8 期間）。

## 4. 入力データ要件（public.loto_hist）

### 4.1 必須カラム

* loto：TEXT
* unique_id：TEXT
* ds：DATE もしくは TIMESTAMP（時系列の軸）
* y：数値（INTEGER 推奨）

### 4.2 推奨カラム（監査・運用）

* exec_ts, updated_ts, proc_seconds 等（スナップショット判定や監査用途）

### 4.3 スナップショット解決

* 同一 (loto, unique_id, ds) の複数行が存在する場合、**最新の1行に解決**してから特徴量生成する。
* 最新判定は exec_ts DESC, updated_ts DESC を優先（両方ある場合）。

## 5. 出力データ要件（public.loto_hist_exog）

### 5.1 主キー・粒度

* 主キー：(loto, unique_id, ds)
* 粒度：入力と同じ（系列×日付）

### 5.2 出力カラム構成

* キーカラム：loto, unique_id, ds
* 参照用：y（任意だが再現性・監査のため保持推奨）
* メタ：created_at、必要なら exec_ts, updated_ts, proc_seconds
* 特徴量カラム群：本要件で定義する hist_exog のみ（将来拡張可能）

### 5.3 欠損の扱い

* ラグ等により先頭側に欠損が必ず発生する。
* 原則：**モデル投入用には欠損を 0 または適切な埋め値に変換**できる設計にする（ただし、監査目的で欠損保持のモードも許容）。
* 採用方針は設定（config）で切替可能にする。

## 6. hist_exog_list（特徴量）要件

### 6.1 大分類（種類）

以下の「種類」を最低限サポートし、実際に採用する窓幅・本数は設定で変更可能とする。

1. **遅延情報**

* ラグ（複数本）
* 季節ラグ（週次なら 52 など）

2. **平滑化**

* ローリング平均
* 指数移動平均（EWM/EMA）

3. **ばらつき**

* ローリング標準偏差 / 分散
* ローリングレンジ（最大−最小）
* ローリング中央値 / 分位点

4. **変化量**

* 差分（1階、k階）
* 変化率（%変化、対数差分）

5. **位置づけ**

* 過去窓内の順位・パーセンタイル
* ロバストスコア（MAD 等）

6. **異常値系（追加要件）**

* 統計ベース異常スコア/フラグ（Z、IQR、MAD）
* 制御図系（CUSUM、EWMA など）スコア/フラグ
* 距離・密度・木モデル等の異常スコア（例：中心距離、LOF、Isolation Forest など）

  * ただし「未来を見ない更新方式」を必須とする（後述）

7. **クラスタリング系（追加要件）**

* 窓要約ベクトル（統計量の束）
* 形状クラスタID（ラベル）
* 所属確率（ソフト割当が可能なら）
* クラスタ中心距離
* 次元圧縮（PCA成分スコア等）

  * こちらも「未来を見ない学習/更新方式」を必須とする（後述）

### 6.2 命名規約

* 形式：<family>_<param>_<stat> を基本とし、人が見て意味が分かる名前にする。
* 例：lag_*, roll*_*, ewm_*, anom_*, cluster_* など。
* 予約語回避・小文字スネークケース推奨。

## 7. リーク防止（最重要要件）

### 7.1 算出ルール

* 時点 t の特徴量は **t より過去の y のみ**で計算する。
* 実装上は「y を 1 期以上シフトした系列」から計算すること（例：shift(1)）。

### 7.2 異常値・クラスタリング特有の制約

* 異常検知・クラスタリングのモデル（閾値、中心、基底など）は以下のいずれかで運用する：

  1. **学習期間で学習し固定**（推論期間では再学習しない）
  2. **オンライン更新**（ただし更新に使うデータはその時点までに限定）
* “全期間で一括学習してから各時点のスコアを計算”は禁止（未来混入）。

## 8. 設定（コンフィグ）要件

* ラグ本数、窓幅、異常検知の手法、クラスタ数等はコード固定にせず、設定で変更可能にする。
* 例：

  * LAGS（配列）
  * ROLLING_WINDOWS（配列）
  * EWM_ALPHA（配列）
  * ANOMALY_METHODS（有効化フラグ）
  * CLUSTER_METHODS（有効化フラグ）

## 9. パイプライン要件

### 9.1 実行形態

* Python スクリプトとして実行可能（例：ds/src/create_loto_hist_exog.py）
* 任意でシェルラッパー提供（例：ds/bin/run_loto_hist_exog.sh）
* Notebook 実行（/mnt/e/env/ts/tslib/test.ipynb）に対応

### 9.2 冪等性

* 同条件で再実行しても同じ結果になること。
* 方式：全件削除→再投入、または UPSERT（設定で選択可）。

### 9.3 ログ

* 実行開始・終了、入力件数、出力件数、特徴量数、所要時間、例外スタックトレースを記録する。
* ログ出力先：ds/logs/loto_hist_exog.log

## 10. 品質要件（検証・監査）

### 10.1 統計チェック（最低限）

* 行数：入力（最新スナップショット解決後）と出力が一致する
* ds 範囲が期待通り
* 欠損率：設計通り（欠損埋めモードなら 0%、保持モードなら先頭のみ欠損など合理的）

### 10.2 リーク検証（必須）

* 任意サンプルに対し、ある時点 t の特徴量計算に t 以降の y が影響しないことを確認する自動テストを用意する。

  * 例：未来部分の y を改変しても t 時点の特徴量が変わらない

### 10.3 異常値・クラスタの妥当性

* 異常フラグの発火率が極端（0% や 100%）になっていないこと
* クラスタが 1 つに潰れていないこと（設定した場合）

## 11. 受入基準

* public.loto_hist_exog が作成され、主キー重複がない
* 設定した hist exog 種類が生成され、列数が仕様と一致
* リーク検証テストに合格
* DB への投入が完了し、ログに成功が記録される

---

必要なら、この要件定義書に合わせて **「採用する窓幅・ラグ一覧（設定案）」**と、**「異常値・クラスタリングをリーク無しで実装する方式（固定学習 vs オンライン更新）」**を具体案として追記した版も作れます。
その他に追加できる外生変数や手法、ライブラリについて網羅的、再帰的、探索的に検索、調査して一覧表で表現してください

## Response:
思考時間: 1m 32s

以下は、あなたの要件（**hist\_exog =「未来を見ない」過去由来特徴量**）に合わせて、追加できる外生変数・手法・ライブラリを“探索的に広く”棚卸しした一覧です。ポイントは一貫してこれです：

*   **時点 t の特徴量は `y_shift = y.shift(1)` から作る**（t の y 自体すら使わない。要件に最も安全寄り）
*   窓統計は **center=False（片側窓）**、異常/クラスタは **「predict → update（オンライン）」** か \*\*「学習期間で固定」\*\*のみ

* * *

追加候補（hist\_exog）・手法・ライブラリ 一覧表
-----------------------------

| カテゴリ | 追加できる特徴量（例） | 典型パラメータ例 | リーク無しの作り方（要点） | 実装候補ライブラリ | 備考 |
| --- | --- | --- | --- | --- | --- |
| NeuralForecast連携 | hist\_exog の受け渡し | `hist_exog_list=[...]` | dfに列を追加し、hist\_exog\_listに列名を渡す | NeuralForecast（Nixtla） [Nixtlaverse+1](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html?utm_source=chatgpt.com) | まずは「列＝特徴量」の設計を固めるのが最優先 |
| 基本：ラグ拡張 | 多段ラグ、季節ラグ、ラグ差分（lag\_k - lag\_{k+s}） | L=1..56, S=7/28/52 | `y_shift` をさらにshiftして差分を作る | pandas/SQL | 季節性が強い系列ほど効く |
| 基本：ローリング高次 | skew/kurtosis、トリム平均、winsor平均 | w=8/16/32 | `y_shift.rolling(w)`で高次統計 | pandas/scipy | 外れ値に強い集約が便利 |
| ローリング回帰 | 窓内の傾き（トレンド係数）、R²、残差分散 | w=16/32 | 窓内で OLS、ただし入力は常に過去のみ | statsmodels | “最近上がり基調”を数値化できる |
| 指数平滑 | EMA、EWMA分散、EWMA z-score | α=0.1/0.3 等 | `y_shift.ewm(alpha=α, adjust=False)` | pandas | 片側なのでリークしにくい |
| 自己相関系 | ACF/PACF（窓内）、季節自己相関 | w=64, lag=7/14 | 窓を切ってから相関算出（毎tは重いのでバッチ/間引き推奨） | statsmodels | 特徴量数を増やしすぎ注意 |
| 分布形状 | 分位点束（p10,p50,p90）、IQR、Gini | w=16/32 | `y_shift.rolling(w).quantile(p)` | pandas | “レンジの歪み”が効くことがある |
| ラン長/間隔 | 連続増加日数、連続ゼロ日数、最後のピークからの経過 | しきい値=0/medianなど | 過去のみで状態を更新（カウンタ） | 自前実装 | 宝くじ系みたいな離散系列で効くことがある |
| スペクトル | FFTパワー、支配周波数、スペクトルエントロピー | w=64/128 | 窓を切ってからFFT（過去窓のみ） | numpy/scipy / tsfeatures(R参考) [CRAN](https://cran.r-project.org/web/packages/tsfeatures/tsfeatures.pdf?utm_source=chatgpt.com) | 計算コスト高め、間引き推奨 |
| エントロピー/複雑度 | permutation/sample entropy、SVD entropy | w=64 | 窓を切って複雑度を算出（過去窓のみ） | antropy [GitHub](https://github.com/raphaelvallat/antropy?utm_source=chatgpt.com) / EntropyHub [EntropyHub](https://www.entropyhub.xyz/python/pyAPI.html?utm_source=chatgpt.com) | ノイズに強い指標も多い |
| 非線形ダイナミクス | Hurst、DFA、Lyapunov | w=256+ | 窓内で算出（過去窓のみ） | nolds [Nolds+1](https://nolds.readthedocs.io/en/latest/nolds.html?utm_source=chatgpt.com) | 長め窓が必要になりがち |
| 変化点検知 | 変化点スコア、直近変化点からの距離 | penalty/モデル | “未来を見ない”なら **オンラインCPD**か、**train固定→過去窓で判定** | ruptures [Centre Borelli+1](https://centre-borelli.github.io/ruptures-docs/?utm_source=chatgpt.com) | ruptures自体はオフライン前提なので運用設計注意 |
| CUSUM/安定性 | CUSUM統計、パラメータ安定性検定値 | 係数モデル | “過去だけの残差”に対してCUSUMを回す | statsmodels [Statsmodels+1](https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html?utm_source=chatgpt.com) | 「レジーム変化」検知として強い |
| 統計ベース異常 | rolling z / IQR / MAD、フラグ | w=16/32 | **基準統計は必ず過去窓**（`y_shift`） | pandas | 最初に入れるべき堅実枠 |
| 行動監視系 | EWMA管理図スコア、CUSUM管理図 | λ, h など | “スコア算出→当日値で更新”の順 | statsmodels（CUSUM周辺） [Statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html?utm_source=chatgpt.com) | 工場監視の定番を時系列特徴へ転用 |
| 木モデル異常（固定） | Isolation Forestスコア | contamination等 | **学習期間だけでfit**→各tは過去窓特徴からscore | scikit-learn [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html?utm_source=chatgpt.com) | “全期間fit”はリークなので禁止 |
| 木モデル異常（オンライン） | HSTスコア（IsolationForestのオンライン版） | n\_trees等 | **score→learn**（predict-then-update） | river HalfSpaceTrees [Riverml+1](https://riverml.xyz/dev/api/anomaly/HalfSpaceTrees/?utm_source=chatgpt.com) | “未来を見ない更新方式”に最も素直 |
| 密度/距離異常 | LOF/OCSVM/ABOD等のスコア | k等 | fitはtrain固定、またはオンライン対応手法を選ぶ | PyOD [PyOD](https://pyod.readthedocs.io/?utm_source=chatgpt.com) / scikit-learn [scikit-learn](https://scikit-learn.org/stable/modules/outlier_detection.html?utm_source=chatgpt.com) | LOFは“novelty設定”など癖あり |
| ドリフト検知 | 分布ドリフトスコア（窓間比較） | w\_short/w\_long | 直近窓 vs 過去窓の距離（KS, PSI等） | Alibi Detect（ドリフト含む） [Seldon](https://docs.seldon.ai/alibi-detect?utm_source=chatgpt.com) | 目的変数yの“質”変化を特徴化 |
| Prophet系外れ値 | 時系列外れ値検知スコア | seasonality | **学習期間でモデル固定**し、過去データのみで更新設計 | Alibi Detect Prophet例 [Seldon](https://docs.seldon.ai/alibi-detect/outlier-detection/examples/od_prophet_weather?utm_source=chatgpt.com) | 便利だが運用設計を誤るとリークしやすい |
| 形状・モチーフ | モチーフ距離、discordスコア | m（サブ系列長） | **過去系列だけ**で最近傍距離を計算 | STUMPY（Matrix Profile） [STUMPY+2STUMPY+2](https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html?utm_source=chatgpt.com) | 「形が違う」を強く拾える（異常にも強い） |
| 形状クラスタ | DTW k-means、Soft-DTW | k, metric | 各tで「過去窓」を1サンプルとしてクラスタリング（train固定 or online更新） | tslearn [tslearn+2tslearn+2](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html?utm_source=chatgpt.com) | 窓を“ベクトル化/系列化”してからがコツ |
| クラスタID/距離 | cluster\_id、中心距離、所属確率 | k | **score→update**（オンラインなら）／train固定ならtransformのみ | scikit-learn MiniBatchKMeans [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html?utm_source=chatgpt.com) | 距離は連続特徴として使いやすい |
| 次元圧縮 | PCA成分スコア（過去窓特徴の圧縮） | n\_comp | train固定、または `partial_fit` で逐次更新 | IncrementalPCA [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html?utm_source=chatgpt.com) | “特徴量爆発”を抑える消火器 |
| GPU高速化 | GPUで特徴/距離/モチーフ | \- | 計算の中身は同じ、入力は過去のみ | Khiva [Khiva+1](https://khiva.readthedocs.io/_/downloads/en/latest/pdf/?utm_source=chatgpt.com) / cuML [RAPIDS Docs+1](https://docs.rapids.ai/api/cuml/stable/?utm_source=chatgpt.com) | 大規模時の現実解 |
| 自動特徴抽出 | 大量の統計・周波数・形状特徴 | presets | 各tで過去窓を作って“窓→特徴ベクトル” | tsfresh [tsfresh+1](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html?utm_source=chatgpt.com) | “efficient/minimal/comprehensive”が選べる [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html?utm_source=chatgpt.com) |
| 自動特徴抽出（軽量） | 統計/周波数/フラクタル特徴 | 65+ | 過去窓→特徴（同上） | TSFEL [TSFEL+1](https://tsfel.readthedocs.io/?utm_source=chatgpt.com) | tsfreshより軽めに回したい時に便利 |
| 自動特徴抽出（別系統） | 汎用特徴セット | \- | 過去窓→特徴 | cesium [Cesium+1](https://cesium-ml.org/docs/?utm_source=chatgpt.com) | “時系列→特徴→ML”の古典的枠組み |
| 柔軟な窓処理 | window/strideで特徴束を組む | stride | 過去窓を明示して特徴を並べる | tsflex [Predict IDLab+1](https://predict-idlab.github.io/tsflex/features/?utm_source=chatgpt.com) | 大量特徴の作業台として便利 |
| 変換テンプレ | “窓要約 + ラグ”をパイプライン化 | \- | `y_shift`を入力に変換器を適用 | sktime transformations [Sktime+1](https://www.sktime.net/en/latest/api_reference/transformations.html?utm_source=chatgpt.com) | Catch22/tsfresh連携も用意されている [sktime+1](https://sktime-backup.readthedocs.io/en/v0.11.4/api_reference/transformations.html?utm_source=chatgpt.com) |
| Catch22特徴 | 代表的な22特徴（軽量） | \- | 過去窓→22特徴 | sktime（Catch22） [sktime](https://sktime-backup.readthedocs.io/en/v0.11.4/api_reference/transformations.html?utm_source=chatgpt.com) | tsfreshよりだいぶ軽い |
| SAX/BoW変換 | SAXで記号化→単語頻度 | window\_size等 | 過去窓を記号列にして頻度/距離にする | pyts [pyts+1](https://pyts.readthedocs.io/en/0.9.0/user_guide.html?utm_source=chatgpt.com) | “形状の言語化”みたいな特徴になる |
| セグメント特徴 | sliding windowで固定長化→特徴 | overlap | “過去窓を切る”処理を標準化 | seglearn [Dmbee+1](https://dmbee.github.io/seglearn/user_guide.html?utm_source=chatgpt.com) | 量産ライン向き |
| 再帰特徴（メタ） | 異常スコアの移動平均、クラスタIDの遷移率 | w | **一次特徴→二次特徴**も必ず過去だけで更新 | 自前/上記全部 | 予測が上手い現場ほどここが効く |
| 時間系の補助（参考） | 曜日/週番号など | \- | 未来既知なので **futr\_exog寄り** | sktime DateTimeFeatures [Sktime](https://www.sktime.net/en/v0.38.4/api_reference/auto_generated/sktime.transformations.series.date.DateTimeFeatures.html?utm_source=chatgpt.com) | 今回スコープ外寄りだが、性能は上がりがち |

* * *

異常値・クラスタリングを「リーク無し」で回す定石（超重要）
-----------------------------

### A. 学習期間で固定（オフライン固定）

*   **train期間の“過去窓”だけ**を集めてモデルをfit
*   推論では **モデルを再学習しない**、各tは `過去窓→特徴→score/cluster` するだけ
*   例：IsolationForest（固定） [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html?utm_source=chatgpt.com)、tslearnクラスタ（固定） [tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html?utm_source=chatgpt.com)

### B. オンライン更新（predict → update）

*   各tで **(1) score/予測 → (2) その時点の情報でlearn/update**
*   Riverの異常検知は `score_one` を持つ設計 [Riverml](https://riverml.xyz/dev/api/overview/?utm_source=chatgpt.com)、HalfSpaceTreesはオンラインIsolationForestの一種 [Riverml](https://riverml.xyz/dev/api/anomaly/HalfSpaceTrees/?utm_source=chatgpt.com)
*   Matrix Profileも“ストリーミング更新”の仕組みがある [STUMPY+1](https://stumpy.readthedocs.io/en/latest/_modules/stumpy/stumpi.html?utm_source=chatgpt.com)
*   クラスタなら MiniBatchKMeans / IncrementalPCA を `partial_fit` で更新 [scikit-learn+1](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html?utm_source=chatgpt.com)

* * *

実務での“おすすめ優先順位”（増やしすぎ事故を防ぐ）
--------------------------

1.  **軽くて堅い**：lag / rolling統計 / EMA / robust（MAD, IQR）
2.  **変化に強い**：変化点（CUSUM系） [Statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html?utm_source=chatgpt.com)、ドリフト/異常スコア（River HST） [Riverml](https://riverml.xyz/dev/api/anomaly/HalfSpaceTrees/?utm_source=chatgpt.com)
3.  **形状が効くなら**：Matrix Profile（STUMPY） [STUMPY](https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html?utm_source=chatgpt.com) or DTWクラスタ（tslearn） [tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html?utm_source=chatgpt.com)
4.  **網羅探索**：tsfresh/TSFELで特徴候補を出してから絞る [tsfresh+1](https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html?utm_source=chatgpt.com)

* * *

必要なら、この表をそのまま \*\*「採用候補 → 採用確定」\*\*に落とすための、実運用向けテンプレ（例：`BASE_FEATURES`, `ANOMALY_FEATURES`, `CLUSTER_FEATURES` を段階的にON/OFFするconfig雛形、リーク検証の自動テスト雛形、特徴量辞書のメタ管理）まで一気に文章化して追記版にできます。

## Prompt:
その他で追加できそうな外生変数や調査漏れがないかファクトチェックして再調査して新規で追加の外生変数を調査してください

## Response:
思考時間: 1m 19s

再調査して棚卸しをやり直したところ、前回の一覧に「有名どころの時系列特徴抽出・異常検知フレームワーク」「分解（トレンド/季節/残差）由来の特徴」「概念ドリフト検知」「shapelet（局所形状）特徴」「別実装のMatrix Profile」「BOCPD系の“オンライン変化点”」あたりが抜けていました。ついでに **Luminaire の出どころは Uber ではなく Zillow** でした（公式リポジトリ/ドキュメントが Zillow 名義）。 [GitHub+1](https://github.com/zillow/luminaire?utm_source=chatgpt.com)

以下、**“新規追加分”として有力な hist\_exog 候補**だけをまとめます（全部「`y_shift = y.shift(1)` から計算」＋「predict→update or 学習固定」でリーク無し運用可能）。

* * *

新規で追加を推奨できる hist\_exog（再調査での追加分）
--------------------------------

| 追加カテゴリ | 追加できる外生変数（例） | リーク無しの実装メモ | 代表ライブラリ |
| --- | --- | --- | --- |
| 総合：特徴量抽出セット（抜けてた） | “定番特徴の詰め合わせ”を一括生成（統計・季節強度など） | 各時点tで「過去窓」を切って特徴化（窓→ベクトル） | **Kats TSFeatures**（65特徴） [Facebook Research+1](https://facebookresearch.github.io/Kats/?utm_source=chatgpt.com) |
| 異常検知（ルール/教師なし系フレームワーク） | level shift / spike / seasonality崩れ等の **異常スコア/フラグ** | ルール検知は基本的に片側窓でOK。学習が要るものは train固定 or predict→update | **ADTK** [Anomaly Detection Toolkit+1](https://adtk.readthedocs.io/en/stable/?utm_source=chatgpt.com) |
| 異常・変化点の統合フレームワーク | 異常スコア校正、CPDスコア、アンサンブルスコア | “全期間fit”は禁止。train固定かオンライン更新で運用 | **Merlion** [Salesforce Open Source+1](https://opensource.salesforce.com/Merlion/v1.0.0/index.html?utm_source=chatgpt.com) |
| 監視向け（異常・予測） | 監視用途の異常スコア/フラグ | 運用は train固定が安全。オンライン更新するなら更新順序を厳密化 | **Luminaire**（Zillow） [GitHub+1](https://github.com/zillow/luminaire?utm_source=chatgpt.com) |
| 軽量な異常検知 + 相関 | 異常スコア、相関（原因候補探索） | 相関も「過去窓」だけで算出 | **Luminol** [GitHub](https://github.com/linkedin/luminol?utm_source=chatgpt.com) |
| 変化点（オンライン） | **BOCPD スコア**、直近変化点からの経過、ランレングス期待値 | “オンライン”が前提なのでリークしにくい（それでも入力は y\_shift） | **bocd** [GitHub+1](https://github.com/y-bar/bocd?utm_source=chatgpt.com) / KatsのBOCPD [Facebook Research](https://facebookresearch.github.io/Kats/api/kats.detectors.bocpd.html?utm_source=chatgpt.com) |
| 変化点（別実装） | BOCPDやGPベースCPDなど | 同上（オンライン前提） | **changepoint** [Redpoll](https://redpoll.ai/blog/changepoint/?utm_source=chatgpt.com) |
| 形状・モチーフ（別系統のMatrix Profile実装） | motif距離、discordスコア、regime境界スコア | 過去窓で計算。ストリーミング更新設計も可能 | **matrixprofile** [GitHub+1](https://github.com/matrix-profile-foundation/matrixprofile?utm_source=chatgpt.com) |
| 分解（トレンド/季節/残差）由来（抜けてた） | `stl_trend`, `stl_season`, `stl_resid`、季節強度、残差の分散 | tまでのデータで分解（片側更新）/ あるいは「過去窓だけ」でSTL | **statsmodels STL/MSTL** [Statsmodels+1](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html?utm_source=chatgpt.com) |
| ボラティリティ（条件付き分散） | `garch_sigma2`（推定分散）、標準化残差、ショック指標 | 金融以外でも“荒れ具合”特徴として使える。fitは過去のみ | **arch**（GARCH等） [Arch](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html?utm_source=chatgpt.com) |
| 非線形ダイナミクス（RQA） | recurrence rate / determinism など **RQA指標** | 過去窓→recurrence plot→RQA | **pyunicorn** [ポツダム気候影響研究所](https://www.pik-potsdam.de/~donges/pyunicorn/api/timeseries/recurrence_plot.html?utm_source=chatgpt.com) |
| 概念ドリフト検知（特徴として強い） | driftスコア/警報フラグ（DDM/EDDM等） | “予測誤差”がなくても、単純な統計変化を入力にする設計が可能 | **river drift（EDDMなど）** [RiverML](https://riverml.xyz/0.11.1/api/drift/EDDM/?utm_source=chatgpt.com) |
| 局所形状（shapelet） | shapelet距離ベクトル、最小距離、トップk距離 | train期間でshapelet学習→推論は距離だけ（安全） | **pyts ShapeletTransform** [pyts+1](https://pyts.readthedocs.io/en/latest/modules/transformation.html?utm_source=chatgpt.com) / **tslearn shapelets** [tslearn Docs](https://tslearn.readthedocs.io/en/latest/user_guide/shapelets.html?utm_source=chatgpt.com) / sktime [Sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.shapelet_transform.ShapeletTransform.html?utm_source=chatgpt.com) |
| 表現学習（超上級：埋め込み特徴） | 過去窓→embedding→そのまま連続特徴 | 学習を train固定にするか、厳密なオンライン更新に限定 | **TS2Vec**（公式実装） [GitHub](https://github.com/zhihanyue/ts2vec?utm_source=chatgpt.com) |

* * *

追加分の「命名」例（要件の命名規約に合わせた案）
------------------------

*   分解系：`stl_p{period}_trend`, `stl_p{period}_season`, `stl_p{period}_resid`（MSTLなら `mstl_p{p1}_{p2}_..._season1` など） [Statsmodels+1](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html?utm_source=chatgpt.com)
*   変化点：`cp_bocpd_hazard_*`, `cp_bocpd_runlen_mean`, `cp_time_since_last` [GitHub+1](https://github.com/y-bar/bocd?utm_source=chatgpt.com)
*   ドリフト：`drift_eddm_score`, `drift_eddm_flag` [RiverML](https://riverml.xyz/0.11.1/api/drift/EDDM/?utm_source=chatgpt.com)
*   shapelet：`shp_k{K}_min_dist`, `shp_id{j}_dist` [pyts+1](https://pyts.readthedocs.io/en/latest/modules/transformation.html?utm_source=chatgpt.com)
*   ボラ：`garch_sigma2`, `garch_std_resid` [Arch](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html?utm_source=chatgpt.com)

* * *

ファクトチェック結果の要点（前回との差分）
---------------------

*   **抜け**：Kats（特徴抽出・BOCPD）、ADTK、Merlion、Luminol、matrixprofile（別実装）、STL/MSTL分解特徴、GARCH系、RQA（recurrence plot）、shapelet系、river drift。 [RiverML+9Facebook Research+9Anomaly Detection Toolkit+9](https://facebookresearch.github.io/Kats/?utm_source=chatgpt.com)
*   **訂正**：Luminaire は Zillow 由来（監視向けTSパッケージ）。 [GitHub+1](https://github.com/zillow/luminaire?utm_source=chatgpt.com)

* * *

この追加分を要件定義書に追記するなら、「分解（STL/MSTL）」「BOCPD/ドリフト」「shapelet」「監視系（ADTK/Merlion/Luminaire）」を **“追加要件としての特徴量ファミリー”** に入れると、探索の抜け漏れがかなり減ります。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
# 外生変数生成要件

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 12/15/2025 8:15:21  
**Updated:** 12/15/2025 8:24:24  
**Exported:** 12/15/2025 8:25:36  
**Link:** [https://chatgpt.com/c/693f4507-b2a8-8322-a0d0-76267d18f31a](https://chatgpt.com/c/693f4507-b2a8-8322-a0d0-76267d18f31a)  

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
9 分で完了しました

こちらがご依頼の調査報告です：

NeuralForecast 用の過去外生変数拡張：追加可能な特徴量・手法・ライブラリ
===========================================

本章では、要件定義書で示された **hist exog\_list** の拡張候補として、時系列解析で広く使われている特徴量や異常検知・クラスタリング手法、関連ライブラリを網羅的に調査した。特徴はデータリークを防ぐためにすべて時点  $t$  以前の情報のみから計算できるように設計する必要がある。

1 採用候補の窓幅・ラグ設定案
---------------

*   **ラグ値（`LAGS`）** – 1期前から長期の周期まで複数の遅延値を採用する。例えば：`[1, 7, 14, 21, 28, 52]`。`1` は直前、`7`は週次、`28` は4週、`52` は年次の季節ラグを表す[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features)。必要に応じて 3 日や 2 四半期など業務上意味のあるラグを追加できる。
*   **ローリング窓長（`ROLLING_WINDOWS`）** – ラグと同様に複数設定する。例：`[3, 7, 14, 28, 52]`。ローリング平均や標準偏差の計算に使用し、短期的な変動と長期的な傾向を同時に捉える[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features)。
*   **指数移動平均の減衰パラメータ（`EWM_ALPHA`）** –  $\alpha\in\{0.1,0.2,0.5,0.8\}$  など数種類を試す。 $\alpha$  が大きいほど最近の観測に重み付けする[risingwave.com](https://risingwave.com/blog/effective-anomaly-detection-in-time-series-using-basic-statistics/#:~:text=Time)。
*   **異常検知・クラスタリング用の学習期間** – データリークを防ぐために学習期間を明示的に設定する。モデルを初期期間（例：過去 2 年間）のデータで学習し、その後は以下の方式で更新する。
    1.  **固定学習（Batch）** – 初期期間で学習した閾値やクラスタ中心を固定し、予測期間中は更新しない。CUSUM や EWMA など統計的手法でも固定パラメータを使用する[en.wikipedia.org](https://en.wikipedia.org/wiki/CUSUM#:~:text=In%20statistical%20quality%20control%20%2C,2)[en.wikipedia.org](https://en.wikipedia.org/wiki/EWMA_chart#:~:text=In%20statistical%20quality%20control%20%2C,most%20distant%20samples%20contribute%20very)。
    2.  **オンライン更新** – モデルの状態を逐次的に更新する。Isolation Forest や Local Outlier Factor はオンライン学習版があり、常に `y` を1期以上シフトしてから更新することでリークを防ぐ。

2 追加できる外生変数・特徴量
---------------

以下の表は、カテゴリ別に追加可能な外生変数およびその説明、利用例、関連文献やライブラリをまとめたものである。表に収まらない詳しい説明は本文中で補足する。

### 2.1 遅延・変化量

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| ラグ特徴 | `lag_k`, `seasonal_lag_k` | `y` を  $k$  期シフトした値。`k` は 1 や 7 など。季節性や依存構造を捕捉[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features)。 | dotData の時系列特徴紹介[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features) |
| 変化量 | `diff_k`, `pct_change_k` | ラグ間の差分や変化率。トレンドの強弱を捉える。 |  |
| 累積合計 | `cumulative_sum` | これまでの合計や平均を計算。例えば販売累積量など。 |  |
| 階差の統計 | `rolling_diff_mean`, `rolling_diff_std` | 差分系列に対しローリング平均や標準偏差を計算することで変動の安定性を把握。 |  |

### 2.2 平滑・ばらつき

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| ローリング統計 | `rolling_mean_w`, `rolling_std_w`, `rolling_median_w`, `rolling_iqr_w` | 窓幅  $w$  の移動平均・標準偏差・中央値・四分位範囲。短期ノイズを除去しトレンドを平滑化[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features)。 |  |
| 指数移動平均 | `ewm_mean_alpha`, `ewm_std_alpha` | 重みが指数的に減衰する移動平均[risingwave.com](https://risingwave.com/blog/effective-anomaly-detection-in-time-series-using-basic-statistics/#:~:text=Time)。 $\alpha$  により過去データへの重み付けを調整。 |  |
| ローリング範囲 | `rolling_range_w` | 窓内の最大値と最小値の差。変動幅を示す。 |  |
| ローリング順位/パーセンタイル | `rolling_rank_w`, `rolling_percentile_w` | 窓内における値の順位やパーセンタイル。高/低水準を相対的に評価。 |  |
| ローリング変動率 | `rolling_cv_w` | 変動係数（標準偏差/平均）など。ばらつきの比率を測定。 |  |

### 2.3 統計・分布特性

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| 中心傾向 | `mean`, `median`, `mode` | 平均・中央値・モードなど基本統計量。TSFELの `calc_mean`, `calc_median`, `hist_mode` に対応[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=calc_mean)。 |  |
| 散布度 | `std`, `variance`, `iqr`, `mad` | 標準偏差、分散、四分位範囲（IQR）、中央値絶対偏差など。TSFELは `interq_range` や `median_abs_deviation` を提供[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html#:~:text=tsfel.feature_extraction.features.interq_range%28signal%29)[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html#:~:text=tsfel.feature_extraction.features.median_abs_deviation%28signal%29)。 |  |
| 偏度・尖度 | `skewness`, `kurtosis` | 分布の非対称性や裾の厚さを表す[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=Computes%20the%20Hurst%20exponent%20of,R%2FS%29%20analysis)。 |  |
| エントロピー | `entropy` | Shannon entropy に基づく乱雑さの尺度[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=entropy%28signal)。 |  |
| ヒストグラム統計 | `ecdf_percentile`, `ecdf_slope` | 経験的累積分布関数から得られる分位や傾き[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=ecdf%28signal)。 |  |
| ベンフォード相関 | `benford_correlation` | 桁数分布との相関で異常や操作を検出（tsfreshが実装）。 |  |
| 線形トレンド | `linear_slope`, `linear_intercept` | 線形回帰の傾きと切片を計算。トレンドの方向や強さを示す。 |  |

### 2.4 自己相関・周期性

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| 自己相関 | `autocorr_lag_k`, `partial_autocorr_lag_k` | k期目の自己相関・偏自己相関。周期性や残差の構造を捉える。TSFELの `autocorr` は初めて 1/e で交差するラグを計算[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=abs_energy)。 |  |
| 周期性強度 | `spectral_entropy`, `spectral_centroid`, `spectral_spread` | フーリエスペクトルの統計値。スペクトルエントロピーは規格化されたスペクトル密度のシャノンエントロピー[feasts.tidyverts.org](https://feasts.tidyverts.org/reference/feat_spectral.html#:~:text=The%20spectral%20entropy%20equals%20the,method)。スプレッドや重心は周波数分布の散布や中心を示す[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=spectral_centroid)。 |  |
| 季節性強度 | `stl_seasonality_strength` | STL分解から得た季節性/トレンド/残差の比率（tsfeaturesが提供）。 |  |
| 周期検出 | `fundamental_frequency`, `max_frequency` | 基底周波数や最大周波数を計算し、繰り返し周期を推定[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=fundamental_frequency)。 |  |
| ウェーブレット特徴 | `wavelet_energy`, `wavelet_entropy`, `wavelet_std`, `wavelet_var` | 連続ウェーブレット変換による各スケールのエネルギーやエントロピー[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=wavelet_abs_mean%28signal%2C%C2%A0fs)。短期・長期の変動を同時に捉える。 |  |

### 2.5 フラクタル・複雑性

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| ハースト指数 | `hurst_exponent` | 再標準化範囲 (R/S) 解析により長期記憶や平均回帰性を測定する[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html#:~:text=tsfel.feature_extraction.features.hurst_exponent%28signal%29)。 $H>0.5$  なら持続性、 $H<0.5$  なら反持続性。 |  |
| Higuchi フラクタル次元 | `higuchi_fractal_dimension` | 時系列のグラフのボックスカウント次元を近似。値が大きいほど曲線が粗雑で複雑である[en.wikipedia.org](https://en.wikipedia.org/wiki/Higuchi_dimension#:~:text=In%20fractal%20geometry%20%2C%20the,3)。 |  |
| 最大フラクタル長 | `maximum_fractal_length` | Higuchi 法で求めた最小スケールでの平均曲線長を計算[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html#:~:text=tsfel.feature_extraction.features.maximum_fractal_length%28signal%29)。 |  |
| Lempel–Ziv 複雑度 | `lempel_ziv_complexity` | 辞書型圧縮に基づく複雑性指標。規則的な列は値が小さく、不規則な列ほど増加する[en.wikipedia.org](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv_complexity#:~:text=The%20Lempel%E2%80%93Ziv%20complexity%20is%20a,the%20shallow%20copy)。 |  |
| Detrended Fluctuation Analysis | `dfa_exponent` | 自己相似性や長期相関を検出する尺度で、非定常な信号にも適用できる[en.wikipedia.org](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis#:~:text=Method%20to%20detect%20power,in%20time%20series)。 |  |
| 多重尺度エントロピー | `multiscale_entropy` | 複数の時間スケールでのエントロピーを計算し、雑音と構造のバランスを評価[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=mse%28signal)。 |  |
| 近似エントロピー/サンプルエントロピー | `approximate_entropy`, `sample_entropy` | 位相空間再構成を用いて信号の規則性と予測可能性を評価。MathWorks は近似エントロピーを「非線形系列の規則性と予測不能性を測る統計量」と説明している[mathworks.com](https://www.mathworks.com/help/predmaint/ref/approximateentropy.html#:~:text=,fluctuations%20over%20a%20time%20series)。 |  |

### 2.6 異常検知・統計的監視

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| Z スコア | `z_score`, `ewma_z_score` | 平均と標準偏差から離れた度合いを測る。異常度として利用[risingwave.com](https://risingwave.com/blog/effective-anomaly-detection-in-time-series-using-basic-statistics/#:~:text=Time)。 |  |
| ロバスト Z スコア | `robust_z_score` | 中央値とMADで計算する改良版で外れ値に強い[rubixml.github.io](https://rubixml.github.io/ML/0.3/anomaly-detectors/robust-z-score.html#:~:text=Robust%20Z)。 |  |
| IQR ベース | `iqr_anomaly_flag` | 四分位範囲で外れ値を検出。 |  |
| CUSUM スコア | `cusum_statistic`, `cusum_flag` | 累積和制御図で平均の変化を検出する手法[en.wikipedia.org](https://en.wikipedia.org/wiki/CUSUM#:~:text=In%20statistical%20quality%20control%20%2C,2)。小さなシフト検出に優れる。 |  |
| EWMA スコア | `ewma_statistic`, `ewma_flag` | 指数移動平均に基づく制御図で、全履歴の重み付き平均をプロットする[en.wikipedia.org](https://en.wikipedia.org/wiki/EWMA_chart#:~:text=In%20statistical%20quality%20control%20%2C,most%20distant%20samples%20contribute%20very)。最近のデータにより高い重みを付与。 |  |
| ARIMA 残差異常 | `arima_residual_z`, `arima_residual_ma` | ARIMA モデルで予測し、残差の異常度を z-score 等で計算[risingwave.com](https://risingwave.com/blog/effective-anomaly-detection-in-time-series-using-basic-statistics/#:~:text=Time)。 |  |
| Isolation Forest | `isolation_forest_score` | ランダムな二分探索木でデータを分割し、分割深さが浅い点を異常とする[en.wikipedia.org](https://en.wikipedia.org/wiki/Isolation_forest#:~:text=Isolation%20Forest%20is%20an%20algorithm,class%20distribution%20or%20target%20value)。線形計算量で高次元にも適用。 |  |
| Local Outlier Factor | `lof_score` | 近傍点の局所密度と比較して外れ値を検出。LOF > 1 なら周囲より低密度で外れ値と判断[en.wikipedia.org](https://en.wikipedia.org/wiki/Local_outlier_factor#:~:text=,lower%20density%20than%20its%20neighbors)。 |  |
| Mahalanobis Distance | `mahalanobis_score` | 多変量正規分布の平均からの距離を共分散行列を考慮して計算し、距離が大きい点を異常とみなす[geeksforgeeks.org](https://www.geeksforgeeks.org/machine-learning/multivariate-anomaly-detection/#:~:text=2,mean%20are%20flagged%20as%20anomalies)。 |  |
| Hotelling’s T² | `hotelling_t2_score` | 多変量 t‐検定に基づく外れ値指標[geeksforgeeks.org](https://www.geeksforgeeks.org/machine-learning/multivariate-anomaly-detection/#:~:text=1,outliers%20in%20normally%20distributed%20data)。 |  |
| クラスタ密度 | `cluster_distance`, `cluster_membership` | 過去のクラスタリング結果を用いて各時点のクラスタ中心までの距離や所属確率を計算。周辺データから離れた点は異常として検知。クラスタ間距離を特徴量として用いる提案もある[stats.stackexchange.com](https://stats.stackexchange.com/questions/288668/clustering-as-dimensionality-reduction#:~:text=,to%20the%20original%20feature%20dimension)。 |  |

### 2.7 クラスタリング・次元圧縮

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| クラスタ所属ID | `cluster_id` | 時系列片を特徴量ベクトルに変換し、k‐means や k‐shape などでクラスタリングし、得られたクラスタラベルを特徴量とする[tmastny.github.io](https://tmastny.github.io/tsrecipes/articles/time-series-clustering.html#:~:text=,as%20a%20dimensionality%20reduction%20technique)。 |  |
| クラスタ中心距離 | `cluster_distance_i` | 各クラスタ中心との距離を特徴量として扱うと高次元データを低次元に射影できる[stats.stackexchange.com](https://stats.stackexchange.com/questions/288668/clustering-as-dimensionality-reduction#:~:text=,to%20the%20original%20feature%20dimension)。 |  |
| クラスタ所属確率 | `cluster_prob_i` | ソフトクラスタリング（例えばGaussian Mixture Model）で得られたクラスタ所属確率。 |  |
| クラスタリング方法 | `kshape`, `dtw_kmeans`, `hierarchical_clustering` | 時系列固有の類似度（DTW, shape-based distance）を用いるクラスタリング。 |  |
| 次元圧縮 | `pca_component_1`, `tsne_component_1` | 主成分分析(PCA)やt-SNEで低次元表現を得る。クラスタリング前の前処理としても使える。 |  |

### 2.8 スペクトル・周波数領域

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| フーリエ変換系 | `fft_energy_band_i`, `fft_max_power`, `fft_bandwidth` | 周波数帯域ごとのエネルギーや最大パワースペクトル密度[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=max_frequency)。周期的な変動の強弱を示す。 |  |
| スペクトル勾配・減少 | `spectral_slope`, `spectral_decrease` | 周波数スペクトルの傾きや減衰量[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=Represents%20the%20amount%20of%20decreasing,of%20the%20spectra%20amplitude)。低周波数寄りか高周波数寄りかを表す。 |  |
| スペクトル歪度・尖度 | `spectral_skewness`, `spectral_kurtosis` | 周波数分布の非対称性やピーク度合いを評価[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=spectral_kurtosis)。 |  |
| スペクトルエントロピー | `spectral_entropy` | 正規化したスペクトル密度のシャノンエントロピー[feasts.tidyverts.org](https://feasts.tidyverts.org/reference/feat_spectral.html#:~:text=The%20spectral%20entropy%20equals%20the,method)。フラットなスペクトルは予測困難でエントロピーが高い。 |  |
| 波形統計 | `spectral_centroid`, `spectral_spread`, `spectral_rolloff`, `spectral_rollon` | スペクトルの重心や広がり、エネルギーが一定割合蓄積される周波数を計算[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=spectral_centroid)。 |  |

### 2.9 イベント・カレンダー要因

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| 時刻要因 | `weekday`, `is_weekend`, `month`, `quarter`, `year`, `holiday_flag` | 曜日や月、四半期、祝日フラグなど日付から導出できるダミー変数。週末や祝日の影響を捉える[dotdata.com](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/#:~:text=Four%20Important%20Types%20of%20Time,Features)。 |  |
| 休日特異性 | `days_until_holiday`, `days_since_holiday` | 次の祝日までの日数や直近の祝日からの経過日数。 |  |
| 営業時間帯 | `hour_of_day`, `is_business_hour` | 時間単位の系列では営業時間かどうかの指標。 |  |
| 天候連携 | `temperature_lag`, `rainfall_lag` | 外部テーブルから気象データを取得し、時系列に同期させてラグ特徴として組み込む。 |  |

### 2.10 その他の手法

| カテゴリ | 特徴量例 | 説明/利点 | 主要参考文献 |
| --- | --- | --- | --- |
| 差分安定性 | `lumpiness`, `stability` | タイル状の非重複ウィンドウで平均や分散を計算し、その分散の分散（lumpiness）と平均の分散（stability）を指標とする[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=Computes%20feature%20of%20a%20time,the%20variance%20of%20the%20variances)。 |  |
| レベル/分散シフト検知 | `max_level_shift`, `max_var_shift`, `max_kl_shift` | 滑動窓間の平均・分散・KLダイバージェンスの最大変化を検出[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=Time%20series%20features%20based%20on,sliding%20windows)。 |  |
| ローカルモチーフ | `motif_entropy` | 平均より大きい・小さいを二値化し、長さ3の語のエントロピーを計算[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=Local%20motifs%20in%20a%20binary,hctsa)。 |  |
| 非線形性検定 | `nonlinearity_statistic` | Lee, White & Granger の非線形性検定統計量を基に非線形の度合いを測定[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=Computes%20a%20nonlinearity%20statistic%20based,when%20the%20series%20is%20linear)。 |  |
| サージ・変化率 | `cusum_change_point_index`, `ewma_change_point_index` | CUSUM や EWMA に基づき変化点位置を出力する特徴。 |  |
| ストレッチ長特徴 | `longest_stretch_above_mean`, `longest_stretch_decreasing` | catch22 の特徴の一部で、平均を上回る連続区間の長さや単調減少区間の長さを測定[time-series-features.gitbook.io](https://time-series-features.gitbook.io/catch22-features/feature-overview-table#:~:text=Feature%20overview%20table)。 |  |
| 低周波パワー | `low_freq_power` | catch22 では周波数 0.1 未満のパワーを全体パワーで割った比率を計算[time-series-features.gitbook.io](https://time-series-features.gitbook.io/catch22-features/feature-overview-table#:~:text=Feature%20overview%20table)。 |  |
| 時間反転非対称性 | `time_reversibility_stat` | 時系列の時間反転に対する非対称性（順序反転での非一致度）を測る[time-series-features.gitbook.io](https://time-series-features.gitbook.io/catch22-features/feature-overview-table#:~:text=Feature%20overview%20table)。 |  |

3 特徴抽出ライブラリ
-----------

複数のオープンソースライブラリが大量の時系列特徴量を自動抽出する機能を提供している。要件定義書の `hist_exog` 生成では、これらライブラリの実装や命名規則を参考にして独自実装するか、一部を取り込むことを検討できる。

| ライブラリ | 主な特徴・機能 | 参考文献・リンク |
| --- | --- | --- |
| **tsfresh** (Python) | 数百種類の統計・分布・スペクトル特徴を自動抽出。ユーザが特徴を選択でき、特徴量選定も内蔵。tsfresh の機能一覧には、絶対エネルギー、線形トレンド、自己相関、エントロピー、ピーク数など多様な指標が含まれる[tsfresh.readthedocs.io](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html#:~:text=Overview%20on%20extracted%20features%EF%83%81)。 | [tsfresh Docs](https://tsfresh.readthedocs.io/en/latest/) |
| **catch22** (C)\*\*/\*\*Python | 7000以上の候補から冗長性を削減して選ばれた22個の特徴。分布形状、極端値の出現時刻、自己相関関数の減衰、低周波パワーなど、直感的で計算効率が高い[aeon-toolkit.org](https://www.aeon-toolkit.org/en/stable/examples/transformations/catch22.html#:~:text=Catch22,Gitbook%20of%20the%20original%20creators)[time-series-features.gitbook.io](https://time-series-features.gitbook.io/catch22-features/feature-overview-table#:~:text=Feature%20overview%20table)。 | catch22 GitBook |
| **TSFEL** (Python) | 統計、時系列、スペクトル、フラクタル領域の65以上の特徴を提供。ハースト指数・Higuchi フラクタル次元・Lempel–Ziv 複雑度・DFA・スペクトル中心・波形エネルギーなどが実装されている[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html#:~:text=abs_energy)[tsfel.readthedocs.io](https://tsfel.readthedocs.io/en/latest/descriptions/modules/tsfel.feature_extraction.html#:~:text=tsfel.feature_extraction.features.hurst_exponent%28signal%29)。 | [TSFEL docs](https://tsfel.readthedocs.io/) |
| **tsfeatures** (R/Python ニクストラ版)**/feasts** | Rob Hyndman らによる時系列特徴抽出パッケージ。STL 分解による季節性・トレンド強度、自己相関、平坦区間数、レベルシフト、ランダム性（lumpiness/stability）などを提供[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=Computes%20feature%20of%20a%20time,the%20variance%20of%20the%20variances)。スペクトルエントロピーをシャノンエントロピーと定義する[feasts.tidyverts.org](https://feasts.tidyverts.org/reference/feat_spectral.html#:~:text=The%20spectral%20entropy%20equals%20the,method)。 | [tsfeatures Documentation](https://pkg.robjhyndman.com/tsfeatures/) |
| **Kats** (Meta/FB) | 時系列分析の総合ライブラリで、予測・異常検知・特徴抽出・多変量分析の機能を含む。TSFeature モジュールは65個以上の特徴を生成し、レベルシフト検知、ロバスト統計検知、CUSUM、STL 分解、自己相関、MACD など多岐にわたる[engineering.fb.com](https://engineering.fb.com/2021/06/21/open-source/kats/#:~:text=What%20it%20is%3A)[facebookresearch.github.io](https://facebookresearch.github.io/Kats/api/kats.tsfeatures.tsfeatures.html#:~:text=Process%20time%20series%20data%20into,feature%20groups%20in%20the%20calculations)。 | [Meta Kats Blog](https://engineering.fb.com/2021/06/21/open-source/kats/) |
| **hctsa** (MATLAB/TSFEL 連携) | 「Highly Comparative Time-Series Analysis」プロジェクト。数千種類の特徴を計算し、各時系列の“特徴的指紋”を作成する。[cran.rstudio.com](https://cran.rstudio.com/web/packages/tsfeatures/refman/tsfeatures.html#:~:text=B,2017)に記載された論文が基礎。 |  |
| **TSFRESH 互換フレーム** | Nixtla の `tsfeatures` (Python) は R版の移植で、NeuralForecast の `tsfeatures` や `utilsforecast` から利用可能。 |  |
| **Others** | `sktime` (scikit-learn 風の時系列分析), `tslearn` (時系列クラスタリング/距離計算), `tsaug` (データ拡張), `pyod` (異常検知), `river` (オンライン学習) など。 |  |

4 データリーク防止策の具体案
---------------

### 4.1 遅延処理

*   **シフト計算を徹底** – あらゆる特徴量計算の前に `y` を1期以上シフトして未来情報の混入を防ぐ（時点  $t$  の特徴は  $t\!-​1$  までの `y` を元に計算）。
*   **ローリング計算の窓** – ローリング統計や移動平均は `rolling(window).apply` などの関数で過去の窓に限定する。窓幅より前の部分では欠損が発生するため、設定で `0` 埋めや外挿を選択できるようにする。

### 4.2 異常検知モデル

*   **固定学習モード** – CUSUM や EWMA など統計的手法は学習期間の平均・標準偏差などの基準値を固定する[en.wikipedia.org](https://en.wikipedia.org/wiki/CUSUM#:~:text=In%20statistical%20quality%20control%20%2C,2)[en.wikipedia.org](https://en.wikipedia.org/wiki/EWMA_chart#:~:text=In%20statistical%20quality%20control%20%2C,most%20distant%20samples%20contribute%20very)。Isolation Forest や LOF も初期期間で学習し、そのモデルを予測期間中は更新しない。
*   **オンライン更新モード** – データが増えるにつれモデルを逐次更新するが、更新に使用するデータは常にシフト済み系列に限定する。例えば Isolation Forest のオンライン版では、過去に学習した木に新しいデータを挿入し、古い木を削除する。
*   **閾値/基準値の再推定** – ロバスト Z スコアや IQR 方式では、新しいデータが追加されるたびに中央値や MAD を更新することができるが、その際も直近時点までのデータのみに基づく。

### 4.3 クラスタリング

*   **学習期間を固定** – クラスタリングは初期期間の窓要約ベクトル（例えば tsfresh や catch22 で抽出した統計量）に対して実施する。得られたクラスタ中心やラベルを予測期間中のデータに適用する。
*   **オンラインクラスタ更新** – 新しいデータが到着したら、`shift` 済みの特徴量のみでクラスタ中心を再計算する。k-means ではオンライン更新法（MiniBatch k-means）、Gaussian Mixture ではオンライン EM 等を利用できる。
*   **距離ベース特徴の計算** – 各時点の特徴ベクトルからクラスタ中心への距離や所属確率を算出し、異常度や外れ値検知に利用する[stats.stackexchange.com](https://stats.stackexchange.com/questions/288668/clustering-as-dimensionality-reduction#:~:text=,to%20the%20original%20feature%20dimension)。距離計算に使う特徴ベクトルは必ず過去から作成されたものを用いる。

5 まとめ
-----

NeuralForecast の `hist_exog_list` を拡張する際は、**過去情報のみから特徴を生成してリークを防ぐ**ことが必須である。基本的な遅延・ローリング統計に加え、本報告で列挙したようなフラクタル指標、スペクトル解析、複雑性指標、統計的監視指標、クラスタリング関連特徴を組み込むことで、モデルの表現力を高めることができる。

また、異常検知やクラスタリングは「初期期間で学習して固定する」方法と「オンライン更新」方法の双方を用意し、設定ファイルで切り替えられるようにすることで運用上の柔軟性を確保できる。TSFEL、tsfresh、catch22、tsfeatures、Kats 等のライブラリは多くの特徴量を実装しているため、実装時の参考や検証用ベンチマークとして有用である。

ご確認のうえ、ご不明な点があればお知らせください。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
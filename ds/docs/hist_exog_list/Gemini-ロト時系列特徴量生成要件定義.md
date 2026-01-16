# ロト時系列特徴量生成要件定義

**Exported:** 12/15/2025 8:25:10  
**Link:** [https://gemini.google.com/app/05b12ca741f7c4c3?hl=ja](https://gemini.google.com/app/05b12ca741f7c4c3?hl=ja)

## Prompt:
\# hist\_exog\_list 要件定義書（NeuralForecast向け：過去外生変数）

\## 1. 目的

\* \`public.loto\_hist\` に保存されている時系列（\`loto, unique\_id, ds, y\`）から、NeuralForecast の \`hist\_exog\_list\` として利用可能な「過去由来の外生変数（hist exogenous）」を生成し、学習・推論に再利用できる形で永続化する。

\* 生成された hist exog は、将来データを参照しない（リークしない）ことを第一要件とする。

\## 2. スコープ

\### 2.1 対象

\* 入力テーブル：\`public.loto\_hist\`

\* 出力テーブル：\`public.loto\_hist\_exog\`（新規作成または再生成）

\* 対象キー：\`(loto, unique\_id, ds)\` 単位の時系列

\### 2.2 非対象

\* \`futr\_exog\_list\`（未来外生変数）の作成（既に \`public.loto\_futr\` で実施済み）

\* モデル学習・予測の実行そのもの

\## 3. 用語

\* \*\*hist\_exog\*\*：予測時点より過去の情報から計算できる外生変数（例：ラグ、移動平均、異常スコア）。

\* \*\*リーク（データリーク）\*\*：時点 t の特徴量計算に t 以降の \`y\` を混入させること。

\* \*\*窓（window）\*\*：ローリング統計・クラスタリングなどで用いる過去区間（例：直近 8 期間）。

\## 4. 入力データ要件（public.loto\_hist）

\### 4.1 必須カラム

\* \`loto\`：TEXT

\* \`unique\_id\`：TEXT

\* \`ds\`：DATE もしくは TIMESTAMP（時系列の軸）

\* \`y\`：数値（INTEGER 推奨）

\### 4.2 推奨カラム（監査・運用）

\* \`exec\_ts\`, \`updated\_ts\`, \`proc\_seconds\` 等（スナップショット判定や監査用途）

\### 4.3 スナップショット解決

\* 同一 \`(loto, unique\_id, ds)\` の複数行が存在する場合、\*\*最新の1行に解決\*\*してから特徴量生成する。

\* 最新判定は \`exec\_ts DESC, updated\_ts DESC\` を優先（両方ある場合）。

\## 5. 出力データ要件（public.loto\_hist\_exog）

\### 5.1 主キー・粒度

\* 主キー：\`(loto, unique\_id, ds)\`

\* 粒度：入力と同じ（系列×日付）

\### 5.2 出力カラム構成

\* キーカラム：\`loto\`, \`unique\_id\`, \`ds\`

\* 参照用：\`y\`（任意だが再現性・監査のため保持推奨）

\* メタ：\`created\_at\`、必要なら \`exec\_ts\`, \`updated\_ts\`, \`proc\_seconds\`

\* 特徴量カラム群：本要件で定義する hist\_exog のみ（将来拡張可能）

\### 5.3 欠損の扱い

\* ラグ等により先頭側に欠損が必ず発生する。

\* 原則：\*\*モデル投入用には欠損を 0 または適切な埋め値に変換\*\*できる設計にする（ただし、監査目的で欠損保持のモードも許容）。

\* 採用方針は設定（config）で切替可能にする。

\## 6. hist\_exog\_list（特徴量）要件

\### 6.1 大分類（種類）

以下の「種類」を最低限サポートし、実際に採用する窓幅・本数は設定で変更可能とする。

1\. \*\*遅延情報\*\*

\* ラグ（複数本）

\* 季節ラグ（週次なら 52 など）

2\. \*\*平滑化\*\*

\* ローリング平均

\* 指数移動平均（EWM/EMA）

3\. \*\*ばらつき\*\*

\* ローリング標準偏差 / 分散

\* ローリングレンジ（最大−最小）

\* ローリング中央値 / 分位点

4\. \*\*変化量\*\*

\* 差分（1階、k階）

\* 変化率（%変化、対数差分）

5\. \*\*位置づけ\*\*

\* 過去窓内の順位・パーセンタイル

\* ロバストスコア（MAD 等）

6\. \*\*異常値系（追加要件）\*\*

\* 統計ベース異常スコア/フラグ（Z、IQR、MAD）

\* 制御図系（CUSUM、EWMA など）スコア/フラグ

\* 距離・密度・木モデル等の異常スコア（例：中心距離、LOF、Isolation Forest など）

\* ただし「未来を見ない更新方式」を必須とする（後述）

7\. \*\*クラスタリング系（追加要件）\*\*

\* 窓要約ベクトル（統計量の束）

\* 形状クラスタID（ラベル）

\* 所属確率（ソフト割当が可能なら）

\* クラスタ中心距離

\* 次元圧縮（PCA成分スコア等）

\* こちらも「未来を見ない学習/更新方式」を必須とする（後述）

\### 6.2 命名規約

\* 形式：\`\<family\>\_\<param\>\_\<stat\>\` を基本とし、人が見て意味が分かる名前にする。

\* 例：\`lag\_\*\`, \`roll\*\_\*\`, \`ewm\_\*\`, \`anom\_\*\`, \`cluster\_\*\` など。

\* 予約語回避・小文字スネークケース推奨。

\## 7. リーク防止（最重要要件）

\### 7.1 算出ルール

\* 時点 t の特徴量は \*\*t より過去の \`y\` のみ\*\*で計算する。

\* 実装上は「\`y\` を 1 期以上シフトした系列」から計算すること（例：\`shift(1)\`）。

\### 7.2 異常値・クラスタリング特有の制約

\* 異常検知・クラスタリングのモデル（閾値、中心、基底など）は以下のいずれかで運用する：

1\. \*\*学習期間で学習し固定\*\*（推論期間では再学習しない）

2\. \*\*オンライン更新\*\*（ただし更新に使うデータはその時点までに限定）

\* “全期間で一括学習してから各時点のスコアを計算”は禁止（未来混入）。

\## 8. 設定（コンフィグ）要件

\* ラグ本数、窓幅、異常検知の手法、クラスタ数等はコード固定にせず、設定で変更可能にする。

\* 例：

\* \`LAGS\`（配列）

\* \`ROLLING\_WINDOWS\`（配列）

\* \`EWM\_ALPHA\`（配列）

\* \`ANOMALY\_METHODS\`（有効化フラグ）

\* \`CLUSTER\_METHODS\`（有効化フラグ）

\## 9. パイプライン要件

\### 9.1 実行形態

\* Python スクリプトとして実行可能（例：\`ds/src/create\_loto\_hist\_exog.py\`）

\* 任意でシェルラッパー提供（例：\`ds/bin/run\_loto\_hist\_exog.sh\`）

\* Notebook 実行（\`/mnt/e/env/ts/tslib/test.ipynb\`）に対応

\### 9.2 冪等性

\* 同条件で再実行しても同じ結果になること。

\* 方式：全件削除→再投入、または UPSERT（設定で選択可）。

\### 9.3 ログ

\* 実行開始・終了、入力件数、出力件数、特徴量数、所要時間、例外スタックトレースを記録する。

\* ログ出力先：\`ds/logs/loto\_hist\_exog.log\`

\## 10. 品質要件（検証・監査）

\### 10.1 統計チェック（最低限）

\* 行数：入力（最新スナップショット解決後）と出力が一致する

\* \`ds\` 範囲が期待通り

\* 欠損率：設計通り（欠損埋めモードなら 0%、保持モードなら先頭のみ欠損など合理的）

\### 10.2 リーク検証（必須）

\* 任意サンプルに対し、ある時点 t の特徴量計算に t 以降の \`y\` が影響しないことを確認する自動テストを用意する。

\* 例：未来部分の \`y\` を改変しても t 時点の特徴量が変わらない

\### 10.3 異常値・クラスタの妥当性

\* 異常フラグの発火率が極端（0% や 100%）になっていないこと

\* クラスタが 1 つに潰れていないこと（設定した場合）

\## 11. 受入基準

\* \`public.loto\_hist\_exog\` が作成され、主キー重複がない

\* 設定した hist exog 種類が生成され、列数が仕様と一致

\* リーク検証テストに合格

\* DB への投入が完了し、ログに成功が記録される

\---

必要なら、この要件定義書に合わせて \*\*「採用する窓幅・ラグ一覧（設定案）」\*\*と、\*\*「異常値・クラスタリングをリーク無しで実装する方式（固定学習 vs オンライン更新）」\*\*を具体案として追記した版も作れます。

その他に追加できる外生変数や手法、ライブラリについて網羅的、再帰的、探索的に検索、調査して一覧表で表現してください

## Response:
> Thinking:
> 
> > Thinking:
> > 
> > **Developing Exogenous Variables**
> > 
> > I'm currently focused on the `hist_exog_list` construction. My goal is to extract relevant past exogenous variables from the `public.loto_hist` data, specifically for use with NeuralForecast. I'm exploring the data and considering which variables will be most predictive. I am working towards a list that captures useful historical trends.
> > 
> > **Exploring More Variables**
> > 
> > I've been poring over the requirements document, specifically the "hist\_exog\_list 要件定義書." My current focus is on identifying potential exogenous variables, methods, and libraries beyond basic statistical features that will meet the no-leakage constraint. I'm compiling a table and writing the explanations in Japanese. I'm considering features like Lag, Smoothing, Variance, Change, Position, Anomaly Detection, and Clustering, with an eye towards compatibility with NeuralForecast. The goal is to capture more useful and predictive information.
> > 
> > **Unpacking Feature Categories**
> > 
> > I'm now diving deep into feature engineering, aiming for an exhaustive approach. I'm exploring time-frequency domain analyses with FFT and wavelets, and considering complexity measures like entropy and fractal dimension. I'm also assessing signal processing features and statistical tests, and evaluating advanced anomaly detection methods like Isolation Forest and LOF for rolling windows. The aim is to build a rich and informative set of exogenous variables.
> > 
> > **Compiling Variable Options**
> > 
> > I'm now cataloging all the potential exogenous variables, grouping them into meaningful categories. This includes advanced clustering methods like rolling K-Means and DBSCAN. Also, I am documenting domain-specific lottery features, like hot/cold numbers and odd/even ratios. Furthermore, I've consolidated libraries: tsfresh, tsfel, kats, cesium, darts, pyod, and tslearn. All of these will make a very full table.
> > 
> > **Defining New Categories**
> > 
> > I've organized potential exogenous variables into seven categories: advanced statistics, information theory/complexity (crucial for lottery randomness), frequency domain features, signal processing elements, advanced anomaly detection techniques, advanced clustering/pattern methods, and domain-specific lottery features. I am also outlining recursive and rolling window approaches to address the no-leakage implementation, highlighting the need for rolling training or incremental learning, especially for models like PCA, autoencoders, and isolation forest.

ご提示いただいた要件定義書は、NeuralForecastでの利用を想定し、リーク防止（Future Leakage Prevention）を最優先事項とした非常に堅牢な設計です。

ご要望に基づき、要件定義書の「6. hist\_exog\_list（特徴量）要件」をさらに拡張・深掘りし、**網羅的・再帰的・探索的**に調査した追加の外生変数、手法、および実装に役立つライブラリを一覧表にまとめました。

特に「ロト・ナンバーズ」というデータの性質（ランダム性が高いが、物理的な偏りや周期性が隠れている可能性がある）を考慮し、\*\*「カオス・複雑性解析」「信号処理」「オンライン機械学習」\*\*の観点を強化しています。

* * *

### 追加可能な hist\_exog\_list 候補一覧表

この表は、単なる統計量を超え、時系列の「形状」「複雑さ」「異常性」を捉えるための手法を分類したものです。すべて\*\*「窓（Window）」または「拡大窓（Expanding Window）」\*\*適用下で、時点  $t$  以前のデータのみを使用することでリークを回避します。

| 大分類 | 中分類・手法 | 具体的な特徴量・内容 | 推奨ライブラリ | 適用時の備考（Loto特有の視点） |
| --- | --- | --- | --- | --- |
| **A. 情報理論・複雑性**<br><br>  <br><br>(Chaos/Complexity) | **エントロピー解析** | **Permutation Entropy** (順列エントロピー)<br><br>  <br><br>**Sample/Approximate Entropy** (近似エントロピー)<br><br>  <br><br>**SVD Entropy** | `antropy`<br><br>  <br><br>`tsfresh`<br><br>  <br><br>`ordpy` | 数字の並びが「どれくらいランダムか」を数値化。機械の癖によるランダム性の低下を検知できる可能性がある。 |
|     | **フラクタル次元** | **Hurst Exponent** (ハースト指数)<br><br>  <br><br>**Higuchi Fractal Dimension** | `nlp`<br><br>  <br><br>`antropy` | 系列が「平均回帰的（ジグザグ）」か「トレンド追随的」かを判定。0.5から乖離する場合、予測可能性が高まる。 |
|     | **Lempel-Ziv複雑性** | LZ Complexity | `antropy` | パターンの繰り返し出現度合い（圧縮可能性）を計測。 |
| **B. 信号処理・周波数**<br><br>  <br><br>(Signal Processing) | **フーリエ変換 (FFT)** | **FFT係数** (実部/虚部/絶対値)<br><br>  <br><br>**パワースペクトル密度 (PSD)**<br><br>  <br><br>**スペクトル重心** | `scipy.signal`<br><br>  <br><br>`tsfel` | ※必ずローリング窓内で計算すること。<br><br>  <br><br>特定の周期（例: 特定の曜日や月）に偏りがあるかを周波数領域で捉える。 |
|     | **ウェーブレット変換** | **CWT係数** (連続ウェーブレット)<br><br>  <br><br>**ウェーブレットエネルギー** | `pywt`<br><br>  <br><br>`tsfresh` | 時間と周波数の両方の特性を捉える。短期的な周期変動の検出に有効。 |
|     | **ピーク検出** | **Number of Peaks** (ピーク数)<br><br>  <br><br>**Mean Distance Between Peaks** | `scipy.signal`<br><br>  <br><br>`tsfresh` | 「当選番号の合計値」などが高い状態が続く頻度などを捉える。 |
| **C. 分布・形状特徴**<br><br>  <br><br>(Distribution) | **高次モーメント** | **Skewness** (歪度)<br><br>  <br><br>**Kurtosis** (尖度) | `scipy.stats`<br><br>  <br><br>`pandas` | 分布の裾の厚さや左右非対称性。当選数字の偏りを検出。 |
|     | **確率質量関数推定** | **Binned Entropy**<br><br>  <br><br>**Histogram Bins** (ヒストグラム分布) | `numpy`<br><br>  <br><br>`tsfel` | 直近 $N$ 回で、どの数値帯（例: 0-9, 10-19...）が多く出ているかの分布ベクトル。 |
| **D. 異常検知・変化点**<br><br>  <br><br>(Anomaly/Change) | **マトリックスプロファイル** | **Matrix Profile** (最近傍距離)<br><br>  <br><br>**Discord** (不一致部分、異常)<br><br>  <br><br>**Motif** (繰り返しパターン) | `stumpy`<br><br>  <br><br>`mass_ts` | **最重要候補**。過去の波形と現在の波形が「どれくらい似ていないか」を距離で定量化。異常な当選パターンの検出に強力。 |
|     | **オンライン異常検知** | **Half-Space Trees**<br><br>  <br><br>**Predictive Anomaly Score** | `river`<br><br>  <br><br>`pysad` | **Riverライブラリ**推奨。モデルを逐次更新しながらスコアを出すため、概念ドリフト（機械の経年劣化等）に対応しつつリークなしで実装可能。 |
| **E. ドメイン特化**<br><br>  <br><br>(Lottery Specific) | **出現間隔 (Inter-arrival)** | **Gap Since Last Occurrence** (該当数字の前回出現からの期間)<br><br>  <br><br>**Hot/Cold Number Score** | `custom`<br><br>  <br><br>(Pandas) | ロト分析の王道。各数字ごとの「ご無沙汰度」を時系列の `y`（当選数字やその集計値）に関連付ける特徴量。 |
|     | **偶奇・和・末尾** | **Odd/Even Ratio** (奇数偶数比)<br><br>  <br><br>**Sum of Digits** (数字の和)<br><br>  <br><br>**Last Digit Stats** | `custom` | `y` そのものではなく、`y` を構成する要素（当選番号リスト）から抽出するメタ特徴量。 |
| **F. 自己回帰・モデル係数**<br><br>  <br><br>(Model Coefficients) | **部分自己相関** | **PACF** (偏自己相関関数) の特定ラグの値 | `statsmodels`<br><br>  <br><br>`tsfresh` | 直接的な相関ではなく、間の影響を除去した純粋な相関関係。 |
|     | **ローリング回帰係数** | 直近 $N$ 期間で `y = ax + b` (xは時間) を解いた時の **Slope (傾き)** と **Intercept** | `scipy.stats` | 局所的なトレンドの勢いを数値化。 |

* * *

### 追加実装に向けた技術的推奨事項

上記の表にある高度な特徴量（異常検知・クラスタリング・カオス解析）を、\*\*「未来を見ない（No Leakage）」\*\*で実装するための具体的なアプローチです。

#### 1\. オンライン学習（Incremental Learning）の採用

要件定義書の「異常値・クラスタリング特有の制約」にある「オンライン更新」を実装する場合、**`River`** というPythonライブラリが最適です。

*   **特徴:** データが1点到着するたびにモデル（K-Means, Isolation Forest, Linear Regression等）を学習・更新し、即座に推論可能です。
*   **メリット:** 常に「その時点までの知識」でモデルが作られるため、**原理的に未来のデータへのリークが発生しません。**
*   **実装イメージ:**
    ```
    from river import cluster
    from river import stream
    # 逐次K-Means (ストリームクラスタリング)
    dbstream = cluster.DBSTREAM(n_clusters=5)
    # ループ処理 (時系列順)
    for x, y in stream.iter_pandas(df_hist):
        # 1. まず予測/スコア算出 (この時点では現在のyを含まない状態でのクラスタ判定も可能)
        cluster_id = dbstream.predict_one(x) 
        # 2. その後にモデル更新 (学習)
        dbstream.learn_one(x)
        # 結果を保存 (hist_exogとして)
        save_feature(cluster_id)
    ```

#### 2\. マトリックスプロファイル (Matrix Profile) の活用

時系列データの「部分列（サブシーケンス）」の類似性を全探索する手法です。`stumpy` ライブラリを使用します。

*   **用途:** 直近の窓（例: 過去10回）の動きが、過去の全歴史の中で「どれくらい珍しいか（異常）」、あるいは「いつの動きに似ているか（類似）」を1次元の距離として出力します。
*   **リーク防止:** `stumpy.stump` は全区間で計算しますが、`stump_incremental` や、単純に過去データのみを対象に計算することで回避可能です。

#### 3\. 特徴量抽出ライブラリ `tsfresh` の Rolling 適用

`tsfresh` は数百種類の特徴量を自動生成しますが、通常はバッチ処理用です。これを時系列予測用に使うには `roll_time_series` 機能を使います。

*   **注意:** 計算コストが非常に高いため、事前に有用な特徴量を選別（Feature Selection）してから、本番のETLパイプラインに組み込むことを推奨します。

* * *

### 次のステップのご提案

この一覧表を基に、要件定義書の「6. hist\_exog\_list」を更新、あるいは別添の「詳細設計書」として定義することが可能です。

**私のほうで実施可能なアクション：**

1.  **`tsfresh` や `River` を用いた具体的な Python 実装コード**（モックアップ）の作成。
2.  上記特徴量の中から、ロト・ナンバーズ予測において**特に計算コスト対効果が高そうなものの選抜**（優先度付け）。
3.  **Matrix Profile (`stumpy`)** を用いた、過去の類似パターン検索機能のコード例示。

ご希望のアクションはありますでしょうか？

## Prompt:
その他で追加できそうな外生変数や調査漏れがないかファクトチェックして再調査して新規で追加の外生変数を調査してください

## Response:
> Thinking:
> 
> > Thinking:
> > 
> > **Discovering Missing Factors**
> > 
> > I'm currently focused on uncovering exogenous variables, specifically for Loto/Numbers data. My latest approach involves investigating factors previously overlooked. This builds on my earlier analysis, aiming for a more complete set of potentially influential external data points for improved forecasting accuracy.
> > 
> > **Probing Deeper Dimensions**
> > 
> > I've been re-evaluating the initial response, conducting a gap analysis to pinpoint unexplored avenues for Loto/Numbers data. My attention has turned toward incorporating physics-inspired features, exploring number transitions through graph theory, and delving into topological data analysis. Deep learning embeddings, plus calendar features (like moon phases), are also on my radar.
> > 
> > **Exploring Numerical Gaps**
> > 
> > I'm now investigating GARCH models, to capture volatility in number gaps and sequence mining to treat numbers as strings (N-grams). I'm searching for related literature using phrases like "GARCH for non-financial time series" and "Time series feature extraction for lottery." Simultaneously, I'm brainstorming new features.
> > 
> > **Mapping Trajectory Dynamics**
> > 
> > I'm presently focusing on topological data analysis and physics-inspired features, with a focus on how series embedding creates a shape with holes or loops. Betti numbers and persistence diagrams are key. I'm also examining the physical machine aspect through the lens of ball mixing and phase space reconstruction, including Lyapunov exponents. Finally, I'm integrating SAX and string mining for pattern analysis.
> > 
> > **Defining Feature Categories**
> > 
> > I'm now integrating new findings into structured categories. I'm focusing on "Cyclic/Calendar" features inspired by local Japanese practices, but am keeping the approach scientific. My primary focus remains on developing a robust framework for addressing these specific features. Currently, I'm working on how to integrate the features into the broader analysis, which is crucial for building a more refined model.

ご提示いただいた要件（NeuralForecast用、リーク厳禁、Lotoデータ）に基づき、既存の調査からさらに一歩踏み込み、\*\*「学術界の最新トレンド」「物理シミュレーション」「グラフ理論」\*\*の観点から、未検討の可能性がある外生変数（hist\_exog）を再調査・発掘しました。

前回の回答が「統計・信号処理」中心だったのに対し、今回は\*\*「データの構造・関係性・潜在表現」\*\*に焦点を当てた追加リストです。

* * *

### 新規追加・再調査した hist\_exog\_list 候補

これらは、単なる数値の変動ではなく、「数値間の関係性」や「背後にある力学系の挙動」を捉えるための高度な特徴量です。

| 大分類 | 中分類・手法 | 具体的な特徴量・内容 | 推奨ライブラリ | Loto/Numbersデータへの適用意義 |
| --- | --- | --- | --- | --- |
| **G. グラフ・ネットワーク理論**<br><br>  <br><br>(Graph/Network) | **可視化グラフ (Visibility Graph)** | **Average Degree** (平均次数)<br><br>  <br><br>**Graph Density** (密度)<br><br>  <br><br>**Clustering Coefficient** | `ts2vg`<br><br>  <br><br>`networkx` | 時系列をグラフ構造に変換し、波形の「凸凹の関係性」を構造化します。周期的な波形とランダムな波形ではグラフのトポロジーが明確に異なります。 |
|     | **遷移確率行列 (Markov)** | **Transition Matrix Eigenvalues** (固有値)<br><br>  <br><br>**PageRank Score** | `networkx`<br><br>  <br><br>`numpy` | 数字  $A$  の次に数字  $B$  が出る確率をグラフの遷移として捉え、直近の「ハブ（出やすい数字）」をPageRank等で特定します。 |
| **H. 位相的データ解析**<br><br>  <br><br>(TDA: Topological Data Analysis) | **パーシステントホモロジー** | **Persistence Entropy**<br><br>  <br><br>**Betti Numbers** (穴の数)<br><br>  <br><br>**Bottleneck Distance** | `giotto-tda`<br><br>  <br><br>`gudhi` | データの「形」を解析します。ノイズに非常に強く、従来の統計量では捉えきれない「微細な周期的構造の崩れ」を検知する最先端手法です。 |
| **I. 物理・力学系埋め込み**<br><br>  <br><br>(Physics/Dynamical Systems) | **時間遅れ埋め込み (Time Delay Embedding)** | **Reconstructed Phase Space Trajectory**<br><br>  <br><br>(位相空間軌道の距離特徴) | `sktime`<br><br>  <br><br>`nolitsa` | 1次元の時系列を多次元の「位相空間」に再構成し、アトラクタ（軌道が収束する形状）の性質を数値化します。カオス的な挙動解析の王道です。 |
|     | **リアプノフ指数 (Lyapunov)** | **Max Lyapunov Exponent** | `nolds` | 初期値のわずかな違いが将来にどれだけ拡大するか（予測不可能性の度合い）を計測します。機械の微細なブレを検知できる可能性があります。 |
| **J. 表現学習・埋め込み**<br><br>  <br><br>(Representation Learning) | **教師なしニューラル特徴** | **Autoencoder Reconstruction Error**<br><br>  <br><br>**Bottleneck Latent Vector** | `PyTorch`<br><br>  <br><br>`TensorFlow` | 直近 $N$ 期間のデータをAutoencoderに入力し、その「圧縮表現（潜在ベクトル）」そのものを特徴量とします。人間が定義できない非線形な特徴を抽出します。 |
|     | **Time2Vec** | **Periodic Activation Features** | `custom` (Keras等) | 周期性を学習可能な関数（sin/cos）として捉え、そのパラメータを特徴量化します。 |
| **K. 乱数検定・暗号理論**<br><br>  <br><br>(Randomness Tests) | **NIST SP 800-22 近似** | **Runs Test P-value**<br><br>  <br><br>**Monobit Test P-value**<br><br>  <br><br>**Serial Correlation** | `arch`<br><br>  <br><br>`statsmodels` | 本来は暗号強度や乱数生成器の品質テストに使われる指標。Lotoの結果が「真のランダム」から一時的に偏った（検定に落ちる）瞬間を狙います。 |
| **L. 計量経済・ボラティリティ**<br><br>  <br><br>(Econometrics) | **条件付き分散 (GARCH)** | **Conditional Volatility** (GARCH  $\sigma _{t2}$ )<br><br>  <br><br>**Residuls** | `arch` | 金融工学の手法。「当たり番号の分散（ばらつき具合）」自体が自己回帰的であると仮定し、次回の「荒れ具合」を予測する特徴量です。 |

* * *

### 追加項目の詳細解説と実装イメージ

#### 1\. 可視化グラフ (Visibility Graph) の採用

時系列データにおける各時点をノードとし、互いに「見通せる（間に高い山がない）」時点同士をエッジで結ぶ手法です。

*   **なぜ有効か:** ランダムウォークと、カオス的時系列では、生成されるグラフの次数分布（Degree Distribution）が決定的に異なります。Lotoの抽選機に物理的な癖がある場合、これがグラフ構造の変化として現れる可能性があります。
*   **実装:** `ts2vg` ライブラリを使えば、Pythonから高速に変換・計算可能です。

#### 2\. 位相的データ解析 (TDA)

時系列をポイントクラウド（点群）として扱い、データが持つ「穴（ループ）」の持続性を測ります。

*   **なぜ有効か:** 平均値や分散が変わらなくても、「データの形状（トポロジー）」が変化しているケースを検知できます。NeuralForecastのようなディープラーニングモデルは、この幾何学的な情報を入力として受け取ることで精度が向上することが報告されています。

#### 3\. 教師なし表現学習 (Rolling Autoencoder)

これは「特徴量設計」をAIに任せるアプローチです。

*   **リーク防止の実装:**
    1.  過去データ（例: 2010年〜2020年）のみでAutoencoderを事前学習（Pre-train）して固定する。
    2.  予測時点  $t$  において、直近窓  $X_{t-w:t}$  をそのAutoencoderに入力し、中間層（Latent Vector）の値を取り出す。
    3.  これを `hist_exog` として予測モデルに入力する。
    *   ※モデルを固定すればリークしません。オンライン学習（River等）と組み合わせるとなお強力です。

#### 4\. NIST乱数検定指標

Lotoは「乱数」を作ろうとしていますが、物理的な抽選機やセット球には偏りがあると言われています。

*   **アプローチ:** 直近100回の当選数字を2進数などに変換し、NIST（米国国立標準技術研究所）が定める乱数検定アルゴリズムの一部（Runs TestやFrequency Test）を実行します。
*   **特徴量:** 検定のP値（確率値）。P値が極端に低い（0.01未満など）期間は、「ランダムではない何かが起きている」シグナルとして扱えます。

* * *

### 要件定義書への統合提案

上記の調査を踏まえ、要件定義書の「6. hist\_exog\_list」または「設定（config）」セクションに、以下のフラグを追加することを推奨します。

1.  **`USE_GRAPH_FEATURES`**: Trueの場合、`networkx`/`ts2vg`を用いて可視化グラフ特徴（次数、密度）を計算する。
2.  **`USE_TDA_FEATURES`**: Trueの場合、`giotto-tda`を用いてパーシステントエントロピーを計算する（※計算コストが高いためデフォルトはFalse推奨）。
3.  **`USE_PHYSICS_FEATURES`**: Trueの場合、位相空間埋め込みによる次元・距離特徴を計算する。
4.  **`USE_RANDOMNESS_TESTS`**: Trueの場合、直近窓に対する乱数検定（Runs Test等）のP値を計算する。

これにより、従来の統計量だけでなく、物理的・幾何学的な視点を取り入れた、より堅牢で説明力の高いデータセット構築が可能になります。

実装に進む場合、まずは計算コストの比較的低い **「G. グラフ理論（可視化グラフ）」** と **「K. 乱数検定」** から導入するのが、Loto予測においては費用対効果が高いと考えられます。



---
Powered by [Gemini Exporter](https://www.ai-chat-exporter.com)
# 外生変数作成方法

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 12/14/2025 19:29:18  
**Updated:** 12/14/2025 19:40:41  
**Exported:** 12/14/2025 19:44:20  
**Link:** [https://chatgpt.com/c/693e917d-979c-8321-9176-4a5c07d125ff](https://chatgpt.com/c/693e917d-979c-8321-9176-4a5c07d125ff)  

## Prompt:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loto</th>
      <th>ds</th>
      <th>unique_id</th>
      <th>y</th>
      <th>exec_ts</th>
      <th>updated_ts</th>
      <th>proc_seconds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bingo5</td>
      <td>2017-04-05</td>
      <td>N1</td>
      <td>1</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>1.440689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bingo5</td>
      <td>2017-04-05</td>
      <td>N2</td>
      <td>10</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>1.440689</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bingo5</td>
      <td>2017-04-05</td>
      <td>N3</td>
      <td>13</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>1.440689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bingo5</td>
      <td>2017-04-05</td>
      <td>N4</td>
      <td>19</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>1.440689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bingo5</td>
      <td>2017-04-05</td>
      <td>N5</td>
      <td>23</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>2025-12-14 19:23:32.596338</td>
      <td>1.440689</td>
    </tr>
  </tbody>
</table>
</div>
futr_exog_list is a parameter in NeuralForecast models that specifies a list of future exogenous columns (external variables known in advance for the forecast horizon). These are features like calendar variables, planned promotions, or weather forecasts that you know ahead of time.

Example usage:

model = NHITS(
    h=12,
    input_size=24,
    futr_exog_list=["trend", "calendar_col"],
    ...
)
When using futr_exog_list, you must provide a futr_df DataFrame containing these columns during prediction.

Want to learn more? These pages may help:

NBEATSx model with exogenous variables
Feature engineering for exogenous variables
Explainability tutorial
dsやyについてfutr_exog_listの外生変数を作成したいのでどのようなものが作成できるかを調査してください

## Response:
思考時間: 1m 21s

`futr_exog_list`に入れられるのは、「予測する時点で _未来のhステップ分まで値が確定している_ 外生変数」です。カレンダーや祝日、（需要なら）販促計画、（電力なら）発電量予測みたいなやつですね。NeuralForecast側も **future / historic / static** を分けて扱い、futureを間違えるとデータリークになるよ、と明記しています。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

あなたのデータ（`ds`,`unique_id`,`y`）前提で「作れる候補」を、**futr\_exog（未来確定）** と **hist\_exog（過去から作る）** に分けて整理します。

* * *

1) `ds`だけから作れる futr\_exog（未来確定で作れるやつ）
-------------------------------------

### カレンダー基本セット（定番だけど強い）

*   `year`, `month`, `quarter`
*   `weekofyear`（週番号）
*   `dayofmonth`（月日）
*   `is_month_start`, `is_month_end`
*   `is_quarter_start`, `is_quarter_end`
*   `days_to_month_end`（月末まで残り日数）
*   `t`（開始日からの通し番号。トレンド項として便利）

NeuralForecastのガイドでも「曜日/月/年みたいなカレンダー変数は長い季節性に有効」と推しています。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

### 周期を滑らかに渡す（sin/cosエンコード）

「月=12は大きい、月=1は小さい」問題を避けるやつです。

*   `sin(2π * weekofyear/52)`, `cos(2π * weekofyear/52)`
*   `sin(2π * month/12)`, `cos(2π * month/12)`

（週次データなら週周期、日次なら年周期など、データ頻度に合わせて分母を選ぶのがコツ）

### 日本の祝日・大型連休フラグ（未来確定）

*   `is_holiday`（祝日か）
*   `is_before_holiday`, `is_after_holiday`（祝日前後）
*   `is_golden_week`, `is_obon`, `is_newyear`（期間フラグ）
*   `days_to_next_holiday` / `days_since_prev_holiday`

祝日判定は `jpholiday` みたいなライブラリで機械的に作れます（内閣府データベース）。 [PyPI](https://pypi.org/project/jpholiday/?utm_source=chatgpt.com)  
※ 祝日は将来の制度変更があるので、長期先をやるなら「祝日データのメンテ」も一緒に考えるのが現実的です。

### 「ビンゴ5特有」に寄せた ds 由来の futr\_exog

ビンゴ5は **毎週水曜日が抽せん日** なので、予測対象が「抽せん回」なら `dow` はほぼ定数になりがちです。 [みずほ銀行+1](https://www.mizuhobank.co.jp/takarakuji/products/bingo5/index.html?utm_source=chatgpt.com)  
その代わり、例えばこういう“週次×月内位置”が作れます。

*   `nth_wed_in_month`（その月の第何水曜か：1〜5）
*   `is_last_wed_of_month`（月最後の水曜か）
*   `is_fiscal_year_end_season`（3月/9月/12月など、期末っぽい月フラグ）
*   `draw_index`（抽せん回の通し番号：`ds`から規則的に付与できるなら未来も確定）

* * *

2) `y`から作れるが futr\_exog にはできないもの（→ hist\_exog にする）
--------------------------------------------------

ここが重要ポイントで、**未来のyを使って作る特徴量は futr\_exog に入れた瞬間リーク**になります。NeuralForecastも「historic/futureを取り違えるとリーク」と警告しています。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

ただし、`y`の“過去”から作って **時点tで確定している** 形にすれば、`hist_exog_list`として使えます。

### 代表例（全部「shiftしてから」作るのが鉄則）

*   `y_lag1`, `y_lag2`, …（ラグ）
*   `rolling_mean_k`, `rolling_std_k`（移動平均・分散）
*   `diff1`（前年差分）
*   `ewm_mean`（指数移動平均）
*   （ロトっぽく）`freq_last_k`（直近k回でその数が出た頻度）や `days_since_hit`（最後に出たのはいつ）  
    ※これは「予測できる」とは限らないけど、特徴量としては作れます。宇宙はしばしば無慈悲にランダムなので。

* * *

3) `unique_id`は futr\_exog というより static exog が自然
------------------------------------------------

`unique_id`（N1〜N5）が「ポジション」や「系列の種類」なら、それ自体は未来に依らず一定なので **static** が向いてます。

*   `static_df`で `unique_id` を one-hot / カテゴリ埋め込み用のID化
*   ポジションごとの取り得る範囲（min/max）などが仕様で決まっているなら、それも static にできる

NeuralForecastは static/historic/future を別枠で扱う設計です。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

* * *

4) 実装の形（作り方の型）
--------------

### futr\_df を作るときのルール

`predict`時は **`unique_id`,`ds` と futr\_exog列だけ**を持つ `futr_df` を渡し、**horizon分の行が系列ごとに揃っている必要**があります。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)  
行が欠けるとエラーになるので、`make_future_dataframe()` を使う導線もよく案内されています。 [GitHub](https://github.com/Nixtla/neuralforecast/issues/1017?utm_source=chatgpt.com)

### ざっくりコード例（雰囲気）

*   futr\_exog（ds由来）を `futr_df` 上で作る
*   y由来は `df` 上で `shift` → `hist_exog_list` に入れる

```python
import numpy as np
import pandas as pd
import jpholiday

df["ds"] = pd.to_datetime(df["ds"])

# --- hist_exog: yから（必ずshiftしてリーク回避） ---
g = df.sort_values("ds").groupby("unique_id")
df["y_lag1"] = g["y"].shift(1)
df["y_roll8_mean"] = g["y"].shift(1).rolling(8).mean().reset_index(level=0, drop=True)

# --- futr_exog: dsから（未来も確定） ---
def add_calendar_feats(x):
    x = x.copy()
    x["month"] = x["ds"].dt.month
    x["weekofyear"] = x["ds"].dt.isocalendar().week.astype(int)
    x["is_month_end"] = x["ds"].dt.is_month_end.astype(int)
    x["is_holiday"] = x["ds"].dt.date.map(jpholiday.is_holiday).astype(int)
    # 週次なら52、日次なら365などに合わせる
    x["sin_woy"] = np.sin(2*np.pi*x["weekofyear"]/52.0)
    x["cos_woy"] = np.cos(2*np.pi*x["weekofyear"]/52.0)
    return x

# futr_dfは nf.make_future_dataframe(df=df) 等で作ってから
# futr_df = add_calendar_feats(futr_df)
```

* * *

まとめ（おすすめ構成）
-----------

*   **futr\_exog\_list**：`ds`由来（週番号/月/祝日/連休/第何水曜…）＋周期(sin/cos)＋通し番号t
*   **hist\_exog\_list**：`y`由来（ラグ・移動統計・頻度系など。shift必須）
*   **stat\_exog\_list**：`unique_id`の性質（N1〜N5の違い）を持たせたいなら static\_df

宝くじは「宇宙の無作為さ」を煮詰めた存在なので、外生変数が効くかは別問題として、\*\*モデルとして正しい作法（未来確定＝futr / 過去由来＝hist）\*\*で組むのがまず勝ち筋です。

## Prompt:
Grok-NeuralForecast! Future Exogenous Variables.md

ファイル

Claude-Creating exogenous variables for NeuralForecast forecasting.md

ファイル

futr\_exog\_list\_gemini.md

ファイル

futr\_exog\_list\_gpt.md

ファイル

その他のモデルにも調査してもらいました
その他追加できるfutr_exog_listの外生変数がないかを網羅的、探索的、再帰的に調査、検索してファクトチェックを行い一覧表で表現してください

## Response:
思考時間: 2m 27s

まず前提だけピン留めします。`futr_exog_list`に入れられるのは「**予測時点で、予測ホライズン全体（hステップ先まで）の値が用意できる外生変数**」です。`predict`では `unique_id, ds` と **futr\_exog列を持つ `futr_df`** を渡し、各系列ごとに **h行そろっている**必要があります。 [Nixtlaverse+1](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)  
また、historic を future に置き間違えると **リーク**になります。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)  
そしてNixtla公式は「外生変数を入れるなら `scaler_type` でスケーラを使う」ことを強く推奨しています（“always use a scaler”）。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)

* * *

どのモデルで `futr_exog_list` が効くか（ファクトチェック）
--------------------------------------

NeuralForecastのモデル表には「Exogenous」列があり、**F=future対応**、H=historic対応、S=static対応、`-`\=非対応が明記されています。 [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)  
例：

*   **F/H/S対応**: NHITS, NBEATSx, BiTCN, TFT, TiDE, TCN, LSTM, GRU … [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)
*   **Fのみ対応**: Autoformer, Informer, FEDformer, TimesNet, VanillaTransformer … [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)
*   **外生なし**（`futr_exog_list`自体が使えない）: NBEATS, NLinear, PatchTST, SOFTS, TSMixer, TimeLLM など（表で `-`） [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html)

* * *

`futr_exog_list`に追加できる外生変数候補（網羅的・探索的リスト）
----------------------------------------

「**dsだけで決まる**」ものは最強（未来が確定）。「**外部データが必要**」なものは、未来分をどう確保するかが勝負です。  
※ `y`由来は原則 future にできない（やるなら `hist_exog_list` 側で shift が鉄則）という注意も、あなたのメモにある通りです。

futr\_exog\_list\_gpt

下表は「作れるか？」を **確定度**（未来で値が確定しているか）でファクトチェックしつつ並べたものです。

| カテゴリ | 外生変数例（列名案） | 作り方/データ源 | 確定度 | “再帰的に増やす”拡張（探索の枝分かれ） |
| --- | --- | --- | --- | --- |
| カレンダー基本 | `year, month, quarter, weekofyear, dayofmonth` | `ds`から抽出 | 確定 | one-hot化、平均との差、前年差（※futureでも計算可な形のみ） |
| 週次前提の位置情報 | `weekofyear` | ISO週 | 確定 | `sin/cos`（下の周期表現へ） |
| 月内の「第n曜日」 | `nth_wed_in_month`（例：第何水曜） | `ds`から算出 | 確定 | `is_1st_wed...is_5th_wed` のダミー化
futr\_exog\_list\_gpt

 |
| 月末系 | `is_month_start, is_month_end, days_to_month_end` | `ds`から算出 | 確定 | 月末近辺だけ強調するpiecewise（例：残り7日以内フラグ）

futr\_exog\_list\_gpt

 |
| 四半期境界 | `is_quarter_start, is_quarter_end` | `ds`から算出 | 確定 | 会計期（4月始まり等）に合わせた“会計四半期”版も作る |
| トレンド | `t` / `trend`（開始日からの通し番号） | `ds-min(ds)` | 確定 | `t^2`, `log1p(t)`、スプライン基底（knot固定ならfutureも可）

Claude-Creating exogenous varia…

 |
| 周期表現（超重要） | `sin_woy, cos_woy` 等 | `sin(2π*周期位置/周期長)` | 確定 | “年周期”“月周期”“週周期”を複数重ねる

futr\_exog\_list\_gpt

 |
| 曜日 | `dayofweek` | `ds.dt.dayofweek` | 確定 | 週次（水曜固定）ならほぼ定数＝効果薄になりがち

Claude-Creating exogenous varia…

 |
| 祝日（日本） | `is_holiday` / `holiday_name` | `jpholiday`等 | 確定（※長期は要メンテ） | 祝日前後、連休長、次の祝日までの日数… [PyPI](https://pypi.org/project/jpholiday/?utm_source=chatgpt.com)

futr\_exog\_list\_gpt

 |
| 祝日ライブラリ代替 | `is_holiday` | `holidays` / `holiday-jp` | 確定（実装依存） | 祝日“種別”までone-hot（建国記念日など） [PyPI+1](https://pypi.org/project/holidays/?utm_source=chatgpt.com) |
| 大型連休 | `is_golden_week` 等 | 期間定義で判定 | ほぼ確定 | 連休の「中日」「前日」など位置情報に分解 [Japan Guide](https://www.japan-guide.com/e/e2282.html?utm_source=chatgpt.com) |
| 年末年始の例外 | `is_year_end` / `is_draw_skipped` | 公式の抽せん日規則 | 確定（規則が公開されている範囲） | 「抽せんがない週」を明示的にフラグ化 [宝くじ公式サイト](https://www.takarakuji-official.jp/kuji/bingo5/feature.html?utm_source=chatgpt.com) |
| 抽せん日ルール | `is_draw_day`（水曜） | BINGO5は毎週水曜（原則） | 確定 | ロト種別が増えるなら「曜日×商品」相互作用 [宝くじ公式サイト+1](https://www.takarakuji-official.jp/special/global/en/bingo5.html?utm_source=chatgpt.com) |
| 抽せん回号 | `draw_index`（通し番号） | 開始日からの連番 | 確定 | `draw_index mod k`（周期仮説の探索）

futr\_exog\_list\_gpt

 |
| “月最後の水曜” | `is_last_wed_of_month` | `ds`から算出 | 確定 | last/2nd last…などに細分化

futr\_exog\_list\_gpt

 |
| 六曜（Rokuyō） | `rokuyo`（大安/仏滅…） | 旧暦換算が必要 | 準確定（計算法依存） | 6値one-hot、または“縁起が良い”の二値化 [Rokuyo](https://rokuyo.org/rokuyo/calculation.html?utm_source=chatgpt.com)

futr\_exog\_list\_gemini

 |
| 給料日・ボーナス仮説 | `is_payday25`, `is_bonus_month` | ルール定義 | 準確定（仮定） | 会社別に違う→一般化し過ぎに注意（探索用）

futr\_exog\_list\_gemini

 |
| 会計期末っぽさ | `is_fiscal_year_end_season` | 月で判定（3月等） | 準確定（仮定） | 業界別期末（9月等）を複数用意

futr\_exog\_list\_gpt

 |
| 天文：月齢 | `moon_phase` | 天文計算/暦API | 確定（計算できれば） | 連続値＋sin/cos化（0〜1循環） |
| 季節：二十四節気 | `solar_term` | 天文暦 | 確定（計算できれば） | 24値one-hot、または季節4区分に粗視化 |
| 日照（行動代理） | `day_length` | 緯度経度＋天文計算 | 確定（場所固定なら） | 都市固定/全国平均などスカラー化 |
| 価格・販促（一般用途） | `promo_flag`, `planned_price` | あなたの計画データ | 確定（社内計画が確定していれば） | “販促強度”を段階化、商品別に分解 |
| 予算・広告出稿 | `ad_spend_plan` | 計画データ | 確定（計画があれば） | チャネル別（TV/検索/SNS）に分割 |
| 天気予報 | `temp_fcst, rain_prob_fcst` | 気象予報データ | 予報（=未来値を持てる） | 予報の“発表時刻（vintage）”でリーク管理（重要） |
| イベント（予定） | `major_event_flag` | 公開スケジュール | 準確定 | 規模/カテゴリ別にone-hot |
| 休業日（システム/市場） | `is_market_closed` 等 | カレンダー | 確定 | “連休長”へ派生 |
| 制度変更（休日法改正など） | `holiday_regime_version` | 公表情報 | 確定（公表後） | 祝日ライブラリの更新とセット運用 [PyPI](https://pypi.org/project/jpholiday/?utm_source=chatgpt.com) |
| “ないもの”のファクトチェック | **BINGO5のキャリーオーバー** | 公式説明 | **存在しない** | 代わりに「当せん金額が回ごとに変動」は別変数として検討 [宝くじ公式サイト+1](https://www.takarakuji-official.jp/kuji/bingo5/introduction2.html?utm_source=chatgpt.com) |

### 補足（でも超重要）

*   `unique_id`（N1〜N5）は未来で変わらないなら **futureではなくstatic** に寄せるのが自然です（`stat_exog_list` + `static_df`）。
    futr\_exog\_list\_gpt
    [Nixtlaverse](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html)
*   `futr_df` は「各 `unique_id` × 未来の `ds` が全部そろってる」必要があり、欠けると例のエラーになります。足りない組み合わせは `make_future_dataframe()` で作れます。 [GitHub+1](https://github.com/Nixtla/neuralforecast/issues/1017?utm_source=chatgpt.com)

* * *

“探索的・再帰的”に増やすための実務ルール（無限に増やさないための柵）
-----------------------------------

あなたのケース（宝くじ）だと、外生変数は「当たりやすくなる魔法」ではなく、**季節性や運用のクセが万一あれば拾うための検出器**です。なので探索は、次の順で機械的にやるのが安全です。

1.  \*\*確定カレンダー（ds由来）\*\*をまず全部：`month/weekofyear/t + sin/cos + 月末/四半期境界`（ここはコスパ最強）
    Claude-Creating exogenous varia…
2.  次に **祝日・連休・年末年始例外**（将来も確定しやすい） [PyPI+1](https://pypi.org/project/jpholiday/?utm_source=chatgpt.com)
3.  それでも改善するなら、**六曜・月齢**みたいな「人間行動っぽい代理変数」を追加（効かなかったら即撤退） [Rokuyo](https://rokuyo.org/rokuyo/calculation.html?utm_source=chatgpt.com)
4.  最後に **予報・計画データ**（運用・データ整備コストが上がるので、効果が見えた時だけ）

この手順でいくと、「増やしすぎて学習が壊れる」事故を避けつつ、ちゃんと網羅的に当たれます。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
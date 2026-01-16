# NeuralForecast futr_exog_list 外生変数 完全ガイド
## 🎯 bingo5データ向け網羅的な外生変数一覧

調査日: 2025年12月14日  
対象モデル: NeuralForecast (NHITS, NBEATS, BiTCN, LSTM等)  
データ: Bingo5抽選結果（weekly Wednesday draws）

---

## 📋 外生変数の分類体系

| カテゴリ | 説明 | 推奨度 | 計算コスト |
|---------|------|--------|-----------|
| **基本カレンダー変数** | 日付から直接抽出可能 | ★★★★★ | 低 |
| **周期的エンコーディング** | Sin/Cos変換による連続性保持 | ★★★★★ | 低 |
| **トレンド特徴量** | 時間経過に伴う変化 | ★★★★☆ | 低 |
| **特別な日フラグ** | 祝日・イベント等 | ★★★★☆ | 中 |
| **フーリエ特徴量** | 複雑な季節性のモデリング | ★★★★☆ | 中 |
| **スプライン特徴量** | 滑らかな非線形トレンド | ★★★☆☆ | 高 |
| **RBF特徴量** | ベル型曲線による周期性 | ★★★☆☆ | 高 |
| **相互作用特徴量** | 複数変数の組み合わせ | ★★★☆☆ | 中 |
| **集約・統計特徴量** | 時間窓ベースの統計量 | ★★★☆☆ | 中 |
| **ドメイン特有変数** | 宝くじ/bingo5特有 | ★★★☆☆ | 低〜中 |

---

## 1️⃣ 基本カレンダー変数（Calendar Features）

### 1.1 時間コンポーネント

| 変数名 | 説明 | 実装例 | 範囲 | 用途 |
|--------|------|--------|------|------|
| `year` | 年 | `df['year'] = df['ds'].dt.year` | 2017〜 | 長期トレンド |
| `month` | 月 | `df['month'] = df['ds'].dt.month` | 1-12 | 月次季節性 |
| `day` | 日 | `df['day'] = df['ds'].dt.day` | 1-31 | 日次パターン |
| `quarter` | 四半期 | `df['quarter'] = df['ds'].dt.quarter` | 1-4 | 四半期季節性 |
| `semester` | 半期 | `df['semester'] = (df['month']-1)//6 + 1` | 1-2 | 半期パターン |
| `week_of_year` | 年内週番号 | `df['week'] = df['ds'].dt.isocalendar().week` | 1-53 | 週次トレンド |
| `week_of_month` | 月内週番号 | 計算式参照 | 1-5 | 月内位置 |
| `day_of_year` | 年内日数 | `df['dayofyear'] = df['ds'].dt.dayofyear` | 1-366 | 年次周期 |
| `day_of_week` | 曜日 | `df['dayofweek'] = df['ds'].dt.dayofweek` | 0-6 | 週次パターン |
| `day_name` | 曜日名 | `df['day_name'] = df['ds'].dt.day_name()` | Mon-Sun | カテゴリカル |

### 1.2 位置・境界フラグ

| 変数名 | 説明 | 実装例 | 値 | 用途 |
|--------|------|--------|------|------|
| `is_month_start` | 月初フラグ | `df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)` | 0/1 | 月初効果 |
| `is_month_end` | 月末フラグ | `df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)` | 0/1 | 月末効果 |
| `is_quarter_start` | 四半期初フラグ | `df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)` | 0/1 | 期初効果 |
| `is_quarter_end` | 四半期末フラグ | `df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)` | 0/1 | 期末効果 |
| `is_year_start` | 年初フラグ | `df['is_year_start'] = df['ds'].dt.is_year_start.astype(int)` | 0/1 | 新年効果 |
| `is_year_end` | 年末フラグ | `df['is_year_end'] = df['ds'].dt.is_year_end.astype(int)` | 0/1 | 年末効果 |
| `is_weekend` | 週末フラグ | `df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)` | 0/1 | 週末効果 |
| `is_weekday` | 平日フラグ | `df['is_weekday'] = (df['dayofweek'] < 5).astype(int)` | 0/1 | 平日効果 |

### 1.3 距離・カウント変数

| 変数名 | 説明 | 実装例 | 範囲 | 用途 |
|--------|------|--------|------|------|
| `days_in_month` | 月の日数 | `df['days_in_month'] = df['ds'].dt.days_in_month` | 28-31 | 月長変動 |
| `days_to_month_end` | 月末までの日数 | `df['days_to_month_end'] = df['days_in_month'] - df['day']` | 0-30 | 月末接近度 |
| `days_since_month_start` | 月初からの日数 | `df['days_since_month_start'] = df['day'] - 1` | 0-30 | 月内経過度 |
| `weeks_in_year` | 年の週数 | `df['weeks_in_year'] = 52 + df['ds'].dt.is_leap_year.astype(int)` | 52-53 | 年長変動 |
| `is_leap_year` | 閏年フラグ | `df['is_leap_year'] = df['ds'].dt.is_leap_year.astype(int)` | 0/1 | 閏年効果 |

---

## 2️⃣ 周期的エンコーディング（Cyclical Encoding）

### 2.1 基本周期変数のSin/Cos変換

**重要**: ニューラルネットワークに最適。12月と1月の近接性を表現可能。

| 変数ペア | 周期 | 実装例 | 推奨度 |
|----------|------|--------|--------|
| `month_sin`, `month_cos` | 12ヶ月 | `df['month_sin'] = np.sin(2*np.pi*df['month']/12)`<br>`df['month_cos'] = np.cos(2*np.pi*df['month']/12)` | ★★★★★ |
| `day_sin`, `day_cos` | 31日 | `df['day_sin'] = np.sin(2*np.pi*df['day']/31)`<br>`df['day_cos'] = np.cos(2*np.pi*df['day']/31)` | ★★★★☆ |
| `dayofweek_sin`, `dayofweek_cos` | 7日 | `df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)`<br>`df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)` | ★★★★★ |
| `dayofyear_sin`, `dayofyear_cos` | 365.25日 | `df['doy_sin'] = np.sin(2*np.pi*df['dayofyear']/365.25)`<br>`df['doy_cos'] = np.cos(2*np.pi*df['dayofyear']/365.25)` | ★★★★★ |
| `week_sin`, `week_cos` | 52週 | `df['week_sin'] = np.sin(2*np.pi*df['week']/52)`<br>`df['week_cos'] = np.cos(2*np.pi*df['week']/52)` | ★★★★☆ |
| `quarter_sin`, `quarter_cos` | 4四半期 | `df['quarter_sin'] = np.sin(2*np.pi*df['quarter']/4)`<br>`df['quarter_cos'] = np.cos(2*np.pi*df['quarter']/4)` | ★★★★☆ |
| `hour_sin`, `hour_cos` | 24時間 | `df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)`<br>`df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)` | ★★★☆☆ |

### 2.2 複合周期変数

| 変数ペア | 説明 | 実装例 | 用途 |
|----------|------|--------|------|
| `time_of_year_sin/cos` | 年内位置 | タイムスタンプベース計算 | 連続的年次周期 |
| `time_of_month_sin/cos` | 月内位置 | タイムスタンプベース計算 | 連続的月次周期 |

**実装例**:
```python
# 年内の連続的位置
timestamp_s = df['ds'].map(pd.Timestamp.timestamp)
year_seconds = 365.25 * 24 * 60 * 60
df['time_of_year_sin'] = np.sin(2 * np.pi * timestamp_s / year_seconds)
df['time_of_year_cos'] = np.cos(2 * np.pi * timestamp_s / year_seconds)
```

---

## 3️⃣ トレンド特徴量（Trend Features）

| 変数名 | 説明 | 実装例 | 用途 |
|--------|------|--------|------|
| `trend` | 線形トレンド | `df['trend'] = range(len(df))` | 単純増加傾向 |
| `trend_from_start` | 開始日からの経過日数 | `df['trend'] = (df['ds'] - df['ds'].min()).dt.days` | 実日数ベース |
| `trend_squared` | 二次トレンド | `df['trend_squared'] = df['trend'] ** 2` | 加速的変化 |
| `trend_log` | 対数トレンド | `df['trend_log'] = np.log1p(df['trend'])` | 減速的成長 |
| `trend_sqrt` | 平方根トレンド | `df['trend_sqrt'] = np.sqrt(df['trend'])` | 初期急成長 |
| `trend_normalized` | 正規化トレンド | `df['trend_norm'] = df['trend'] / df['trend'].max()` | 0-1スケール |

---

## 4️⃣ 特別な日フラグ（Special Day Flags）

### 4.1 日本の祝日・イベント

| 変数名 | 説明 | 実装例 | ライブラリ |
|--------|------|--------|-----------|
| `is_holiday` | 国民の祝日 | `import jpholiday`<br>`df['is_holiday'] = df['ds'].apply(lambda x: jpholiday.is_holiday(x)).astype(int)` | jpholiday |
| `is_before_holiday` | 祝日前日 | `df['is_before_holiday'] = df['is_holiday'].shift(-1, fill_value=0)` | - |
| `is_after_holiday` | 祝日翌日 | `df['is_after_holiday'] = df['is_holiday'].shift(1, fill_value=0)` | - |
| `is_golden_week` | ゴールデンウィーク | 4/29-5/5期間判定 | - |
| `is_obon` | お盆期間 | 8/13-8/16期間判定 | - |
| `is_new_year_holiday` | 年末年始 | 12/29-1/3期間判定 | - |
| `is_silver_week` | シルバーウィーク | 9月連休判定 | - |

### 4.2 経済・社会イベント

| 変数名 | 説明 | 実装例 | 用途 |
|--------|------|--------|------|
| `is_payday` | 給料日（25日） | `df['is_payday'] = (df['day'] == 25).astype(int)` | 購買力変化 |
| `is_bonus_month` | ボーナス月 | `df['is_bonus_month'] = df['month'].isin([6, 12]).astype(int)` | 資金余裕 |
| `days_to_next_holiday` | 次の祝日までの日数 | 計算式参照 | 期待感 |
| `days_since_last_holiday` | 前回祝日からの日数 | 計算式参照 | 休日サイクル |
| `is_fiscal_year_end` | 年度末月 | `df['is_fiscal_year_end'] = (df['month'] == 3).astype(int)` | 決算期 |

### 4.3 その他文化的要素

| 変数名 | 説明 | 実装例 | 文化的背景 |
|--------|------|--------|-----------|
| `rokuyo` | 六曜（大安・仏滅等） | `import jpholiday`で取得可能 | 縁起担ぎ |
| `is_taian` | 大安フラグ | 六曜から抽出 | 購買行動影響 |
| `is_butsumetsu` | 仏滅フラグ | 六曜から抽出 | 購買行動影響 |

---

## 5️⃣ フーリエ特徴量（Fourier Features）

複数の季節性を同時にモデリング。高周波パターンの捕捉に有効。

### 5.1 基本実装

```python
from utilsforecast.feature_engineering import fourier

# 週次季節性（k=2: 2つのフーリエ項）
df_transformed, futr_df = fourier(
    df, 
    freq='W-WED',  # 水曜日週次
    season_length=52,  # 年次周期
    k=2,  # フーリエ次数
    h=12  # 予測期間
)
```

### 5.2 カスタムフーリエ項

| 変数名 | 説明 | 実装例 | 周期 |
|--------|------|--------|------|
| `fourier_year_sin_1` | 年次1次Sin | `np.sin(2*np.pi*1*t/52)` | 年1回 |
| `fourier_year_cos_1` | 年次1次Cos | `np.cos(2*np.pi*1*t/52)` | 年1回 |
| `fourier_year_sin_2` | 年次2次Sin | `np.sin(2*np.pi*2*t/52)` | 年2回 |
| `fourier_year_cos_2` | 年次2次Cos | `np.cos(2*np.pi*2*t/52)` | 年2回 |
| `fourier_quarter_sin` | 四半期Sin | `np.sin(2*np.pi*t/13)` | 四半期 |
| `fourier_quarter_cos` | 四半期Cos | `np.cos(2*np.pi*t/13)` | 四半期 |

**次数kの選択**:
- k=1-2: 単純な季節性
- k=3-5: 中程度の複雑性
- k=6以上: 複雑な多重季節性

---

## 6️⃣ スプライン特徴量（Spline Features）

滑らかな非線形トレンドのモデリング。

```python
from scipy.interpolate import BSpline
import numpy as np

# B-スプライン基底関数
def create_bspline_features(df, n_knots=5, degree=3):
    """
    B-スプライン特徴量の生成
    """
    t = np.arange(len(df))
    knots = np.linspace(0, len(df)-1, n_knots)
    
    # 拡張ノット配列
    knots_ext = np.concatenate([
        [knots[0]] * degree,
        knots,
        [knots[-1]] * degree
    ])
    
    # 基底関数の評価
    for i in range(len(knots_ext) - degree - 1):
        coeffs = np.zeros(len(knots_ext) - degree - 1)
        coeffs[i] = 1
        spl = BSpline(knots_ext, coeffs, degree)
        df[f'spline_{i}'] = spl(t)
    
    return df
```

| パラメータ | 説明 | 推奨値 | 効果 |
|-----------|------|--------|------|
| n_knots | ノット数 | 5-10 | 曲線の複雑さ |
| degree | 次数 | 3 (cubic) | 滑らかさ |

---

## 7️⃣ RBF特徴量（Radial Basis Function Features）

ベル型曲線による周期的パターンのモデリング。

```python
from sklego.preprocessing import RepeatingBasisFunction

# 月次パターンのRBF
rbf_encoder = RepeatingBasisFunction(
    n_periods=4,  # 4つのRBF
    column='month',
    input_range=(1, 12),
    width=0.25  # 曲線の幅
)

rbf_features = rbf_encoder.fit_transform(df[['month']])
```

| パラメータ | 説明 | 推奨値 | 効果 |
|-----------|------|--------|------|
| n_periods | RBF数 | 4-12 | 解像度 |
| width | 曲線幅 | 0.1-0.5 | 重なり度 |
| input_range | 入力範囲 | (1,12), (1,7)等 | 対象周期 |

---

## 8️⃣ 相互作用特徴量（Interaction Features）

### 8.1 時間変数間の相互作用

| 変数名 | 説明 | 実装例 | 用途 |
|--------|------|--------|------|
| `month_x_week` | 月×週 | `df['month_week'] = df['month'] * df['week']` | 月内週パターン |
| `quarter_x_month` | 四半期×月 | `df['quarter_month'] = df['quarter'] * df['month']` | 四半期内月 |
| `is_holiday_x_weekend` | 祝日×週末 | `df['holiday_weekend'] = df['is_holiday'] * df['is_weekend']` | 連休効果 |
| `month_x_is_month_end` | 月×月末 | `df['month_end_month'] = df['month'] * df['is_month_end']` | 月末×月 |

### 8.2 多項式特徴量

```python
from sklearn.preprocessing import PolynomialFeatures

# 2次多項式特徴量
poly = PolynomialFeatures(degree=2, include_bias=False)
time_features = df[['month', 'week', 'dayofweek']]
poly_features = poly.fit_transform(time_features)
```

---

## 9️⃣ 集約・統計特徴量（Aggregated/Statistical Features）

### 9.1 移動窓ベース統計量（未来確定の場合のみ）

**注意**: これらは過去データから計算されるため、通常はhist_exog_listに分類されますが、
外部データソースから未来値が確定している場合のみfutr_exog_listに含められます。

| 変数名 | 説明 | 実装例 | 窓サイズ |
|--------|------|--------|----------|
| `rolling_mean_draws` | 直近の抽選回数平均 | 外部データ必要 | 4,8,12週 |
| `jackpot_amount` | キャリーオーバー額 | 事前公表値 | - |
| `prize_pool_size` | 賞金総額 | 事前公表値 | - |

### 9.2 カレンダーベース集約

| 変数名 | 説明 | 実装例 | 用途 |
|--------|------|--------|------|
| `business_days_in_month` | 月の営業日数 | `np.busday_count(...)` | 購買機会 |
| `weekdays_until_month_end` | 月末までの平日数 | 計算式参照 | 給与サイクル |
| `holidays_in_month` | 月内祝日数 | カウント集計 | 休日密度 |

---

## 🔟 ドメイン特有変数（Domain-Specific: Bingo5/宝くじ）

### 10.1 Bingo5抽選特性

| 変数名 | 説明 | 実装例 | 根拠 |
|--------|------|--------|------|
| `draw_number` | 抽選回号 | 連番付与 | 回次トレンド |
| `nth_wednesday_of_month` | 月内第何水曜 | 計算式参照 | 月内位置 |
| `is_last_wednesday` | 月最終水曜フラグ | 判定式 | 月末効果 |
| `draws_this_month` | 月内抽選回数 | カウント | 月長変動 |
| `is_first_draw_of_month` | 月初回抽選 | フラグ | 新規購買 |
| `is_last_draw_of_month` | 月最終抽選 | フラグ | 月末購買 |

### 10.2 時系列パターン特徴（研究ベース）

**注意**: 以下は宝くじ予測研究で使用された特徴量ですが、ランダム性により効果は限定的です。

| 変数名カテゴリ | 説明 | 実装難易度 | 科学的根拠 |
|--------------|------|-----------|-----------|
| 頻度解析系 | Hot/Cold/Overdue数 | 中 | 低（ギャンブラーの誤謬） |
| ギャップ解析 | 出現間隔統計 | 中 | 低 |
| クラスター分析 | 数字の共起パターン | 高 | 低 |
| 物理的バイアス | 機械特性由来の偏り | 極高 | 中（機械依存） |

**実装例（参考）**:
```python
# これらはhist_exog_list向け（過去データ依存）
# 頻度カウント
df['hot_numbers_count'] = df.groupby('unique_id')['y'].rolling(50).apply(
    lambda x: (x > x.median()).sum()
).reset_index(0, drop=True)

# 出現ギャップ
df['gap_since_last'] = df.groupby('unique_id')['y'].diff()
```

### 10.3 外部環境変数（オプション）

| 変数名 | 説明 | データソース | 因果関係 |
|--------|------|-------------|----------|
| `stock_market_index` | 株価指数 | 金融API | 弱（景気感） |
| `consumer_confidence` | 消費者信頼感指数 | 政府統計 | 弱（購買意欲） |
| `unemployment_rate` | 失業率 | 労働統計 | 弱（可処分所得） |
| `weather_condition` | 天候 | 気象API | 極弱 |

**重要**: これらは宝くじ番号予測には科学的根拠が薄く、購買行動予測には有用な可能性があります。

---

## 📊 推奨構成例

### 構成A: ミニマル（計算コスト最小）

```python
futr_exog_list = [
    # 周期的エンコーディング
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    
    # トレンド
    'trend',
    
    # 特別な日
    'is_holiday',
    'is_weekend'
]
```
**特徴**: 10変数、高速、安定

### 構成B: バランス（推奨）

```python
futr_exog_list = [
    # 基本カレンダー
    'month', 'week', 'quarter',
    
    # 周期的エンコーディング
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'week_sin', 'week_cos',
    'quarter_sin', 'quarter_cos',
    
    # トレンド
    'trend', 'trend_squared',
    
    # フラグ
    'is_holiday', 'is_weekend',
    'is_month_start', 'is_month_end',
    'is_quarter_end',
    
    # Bingo5特有
    'nth_wednesday_of_month',
    'is_last_wednesday',
    
    # 距離
    'days_to_month_end'
]
```
**特徴**: 25変数、精度とコストのバランス

### 構成C: フル（最大精度追求）

```python
futr_exog_list = [
    # 基本カレンダー（全て）
    'year', 'month', 'day', 'quarter', 'week',
    'dayofweek', 'dayofyear',
    
    # 周期的エンコーディング（全周期）
    'month_sin', 'month_cos',
    'day_sin', 'day_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'week_sin', 'week_cos',
    'quarter_sin', 'quarter_cos',
    
    # トレンド（複数次数）
    'trend', 'trend_squared', 'trend_log',
    
    # フーリエ項（k=3）
    'fourier_year_sin_1', 'fourier_year_cos_1',
    'fourier_year_sin_2', 'fourier_year_cos_2',
    'fourier_year_sin_3', 'fourier_year_cos_3',
    
    # RBF特徴量（月次）
    'rbf_month_1', 'rbf_month_2', 'rbf_month_3', 'rbf_month_4',
    
    # フラグ（全て）
    'is_holiday', 'is_weekend', 'is_before_holiday', 'is_after_holiday',
    'is_month_start', 'is_month_end',
    'is_quarter_start', 'is_quarter_end',
    'is_golden_week', 'is_obon', 'is_new_year_holiday',
    'is_payday', 'is_bonus_month',
    
    # Bingo5特有（全て）
    'draw_number', 'nth_wednesday_of_month',
    'is_first_draw_of_month', 'is_last_draw_of_month',
    'is_last_wednesday', 'draws_this_month',
    
    # 距離・カウント
    'days_to_month_end', 'days_since_month_start',
    'days_to_next_holiday', 'days_since_last_holiday',
    'holidays_in_month',
    
    # 相互作用
    'month_x_week', 'is_holiday_x_weekend'
]
```
**特徴**: 50+変数、最高精度、高計算コスト

---

## 💻 実装コード集

### 完全実装関数

```python
import pandas as pd
import numpy as np
import jpholiday
from scipy.interpolate import BSpline
from sklego.preprocessing import RepeatingBasisFunction

def create_comprehensive_futr_exog(df, config='balanced'):
    """
    包括的な外生変数生成関数
    
    Parameters:
    -----------
    df : pd.DataFrame
        'ds'列を含むデータフレーム
    config : str
        'minimal', 'balanced', 'full'のいずれか
    
    Returns:
    --------
    pd.DataFrame
        外生変数が追加されたデータフレーム
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # === 1. 基本カレンダー変数 ===
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['quarter'] = df['ds'].dt.quarter
    df['week'] = df['ds'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['dayofyear'] = df['ds'].dt.dayofyear
    
    # === 2. 周期的エンコーディング ===
    # 月（12ヶ月周期）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 曜日（7日周期）
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 年内日数（365.25日周期）
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    if config in ['balanced', 'full']:
        # 週（52週周期）
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # 四半期（4四半期周期）
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    if config == 'full':
        # 日（31日周期）
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # === 3. トレンド特徴量 ===
    base_date = df['ds'].min()
    df['trend'] = (df['ds'] - base_date).dt.days
    
    if config in ['balanced', 'full']:
        df['trend_squared'] = df['trend'] ** 2
    
    if config == 'full':
        df['trend_log'] = np.log1p(df['trend'])
    
    # === 4. 特別な日フラグ ===
    # 祝日
    df['is_holiday'] = df['ds'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    if config in ['balanced', 'full']:
        df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
        
        # Bingo5特有
        df['nth_wednesday_of_month'] = ((df['day'] - 1) // 7) + 1
        df['is_last_wednesday'] = (
            (df['dayofweek'] == 2) & 
            ((df['day'] + 7) > df['ds'].dt.days_in_month)
        ).astype(int)
    
    if config == 'full':
        df['is_before_holiday'] = df['is_holiday'].shift(-1, fill_value=0)
        df['is_after_holiday'] = df['is_holiday'].shift(1, fill_value=0)
        df['is_quarter_start'] = df['ds'].dt.is_quarter_start.astype(int)
        
        # 大型連休
        df['is_golden_week'] = (
            (df['month'] == 4) & (df['day'] >= 29) |
            (df['month'] == 5) & (df['day'] <= 5)
        ).astype(int)
        
        df['is_obon'] = (
            (df['month'] == 8) & (df['day'] >= 13) & (df['day'] <= 16)
        ).astype(int)
        
        df['is_new_year_holiday'] = (
            (df['month'] == 12) & (df['day'] >= 29) |
            (df['month'] == 1) & (df['day'] <= 3)
        ).astype(int)
        
        df['is_payday'] = (df['day'] == 25).astype(int)
        df['is_bonus_month'] = df['month'].isin([6, 12]).astype(int)
        
        # Bingo5追加
        df['draw_number'] = range(len(df))
        df['is_first_draw_of_month'] = (df['day'] <= 7).astype(int)
        df['is_last_draw_of_month'] = (
            (df['ds'] + pd.Timedelta(days=7)).dt.month != df['month']
        ).astype(int)
    
    # === 5. 距離・カウント変数 ===
    if config in ['balanced', 'full']:
        df['days_to_month_end'] = df['ds'].dt.days_in_month - df['day']
    
    if config == 'full':
        df['days_since_month_start'] = df['day'] - 1
        df['days_in_month'] = df['ds'].dt.days_in_month
        
        # 祝日までの距離
        holidays = df[df['is_holiday'] == 1]['ds'].values
        df['days_to_next_holiday'] = df['ds'].apply(
            lambda x: min([abs((pd.Timestamp(h) - x).days) 
                          for h in holidays if pd.Timestamp(h) > x] or [999])
        )
        df['days_since_last_holiday'] = df['ds'].apply(
            lambda x: min([abs((x - pd.Timestamp(h)).days) 
                          for h in holidays if pd.Timestamp(h) < x] or [999])
        )
    
    # === 6. フーリエ特徴量 ===
    if config == 'full':
        t = np.arange(len(df))
        for k in range(1, 4):  # k=1,2,3
            df[f'fourier_year_sin_{k}'] = np.sin(2 * np.pi * k * t / 52)
            df[f'fourier_year_cos_{k}'] = np.cos(2 * np.pi * k * t / 52)
    
    # === 7. RBF特徴量 ===
    if config == 'full':
        try:
            rbf_encoder = RepeatingBasisFunction(
                n_periods=4,
                column='month',
                input_range=(1, 12),
                width=0.25,
                remainder='drop'
            )
            rbf_features = rbf_encoder.fit_transform(df[['month']])
            for i in range(rbf_features.shape[1]):
                df[f'rbf_month_{i+1}'] = rbf_features[:, i]
        except:
            pass  # sklego未インストール時はスキップ
    
    # === 8. 相互作用特徴量 ===
    if config == 'full':
        df['month_x_week'] = df['month'] * df['week']
        df['is_holiday_x_weekend'] = df['is_holiday'] * df['is_weekend']
    
    return df

# === 使用例 ===
# df_minimal = create_comprehensive_futr_exog(df, config='minimal')
# df_balanced = create_comprehensive_futr_exog(df, config='balanced')
# df_full = create_comprehensive_futr_exog(df, config='full')
```

---

## 🔬 科学的考察

### 宝くじ予測における外生変数の限界

**重要な真実**: Bingo5を含む宝くじは物理的ランダム性を持つため、
純粋な数字予測における外生変数の効果は**極めて限定的**です。

#### なぜ外生変数が重要か（モデリング観点）

1. **時系列構造の学習**: モデルが時間的パターンを認識する補助
2. **ベースライン性能向上**: ランダムベースラインを超える精度
3. **汎化性能の改善**: 過学習の抑制
4. **ドメイン知識の組み込み**: データ駆動+知識融合

#### 科学的に有効な応用領域

- 🎯 **購買行動予測**: 販売数、購入者数の予測
- 🎯 **キャリーオーバー予測**: 当選者数、繰越額
- 🎯 **異常検知**: 機械的バイアスの検出
- ❌ **数字予測**: 効果は理論上ゼロ（完全ランダムの場合）

---

## 📚 参考文献・情報源

### 技術文献
1. Nixtla - Exogenous Variables Guide
2. skforecast - Cyclical Features Documentation
3. scikit-lego - Repeating Basis Functions
4. Hyndman & Athanasopoulos - Forecasting: Principles and Practice

### 研究論文
1. Fourier Transform in Time Series (Andrei, 2023)
2. RBF vs Fourier for Seasonality (NVIDIA Technical Blog, 2022)
3. TimeXer: Exogenous Variables in Transformers (2024)

### データソース
- jpholiday: 日本の祝日データ
- utilsforecast: フーリエ変換ユーティリティ

---

## ⚙️ 実装時の注意事項

### 1. スケーリングの必須化

```python
# NeuralForecastでは必須
model = NHITS(
    h=12,
    futr_exog_list=[...],
    scaler_type='robust'  # 必須！
)
```

### 2. futr_dfの完全性

```python
# 予測期間全体をカバー
futr_df = pd.DataFrame({
    'unique_id': np.repeat(df['unique_id'].unique(), h),
    'ds': pd.date_range(
        start=df['ds'].max() + pd.Timedelta(weeks=1),
        periods=h,
        freq='W-WED'
    )
})

# 外生変数を全て追加
futr_df = create_comprehensive_futr_exog(futr_df, config='balanced')
```

### 3. 一貫性の保持

```python
# 訓練時とfutr_df で同じ変数セット
assert set(df.columns) == set(futr_df.columns), "Column mismatch!"
```

### 4. メモリ管理

- Full構成: ~50変数 → メモリ使用量大
- Balanced構成: ~25変数 → 推奨
- Minimal構成: ~10変数 → 低メモリ環境

---

## 🎓 まとめ

### ✅ 推奨アプローチ

1. **開始**: Minimal構成でベースライン確立
2. **実験**: Balanced構成で精度向上を検証
3. **最適化**: 個別変数の重要度分析（SHAP等）
4. **選択**: 有効な変数のみ残してFull構成を構築

### 📈 期待される効果

| 構成 | 変数数 | 訓練時間 | 予測精度期待値 | 推奨用途 |
|------|--------|---------|---------------|----------|
| Minimal | 10 | 1x | Baseline | 試作・検証 |
| Balanced | 25 | 2-3x | +5-10% | 本番推奨 |
| Full | 50+ | 5-10x | +10-15% | 研究・最適化 |

### ⚠️ 最終警告

**宝くじの数字予測において、どれほど精巧な外生変数を用いても、
物理的ランダム性を超える予測は理論上不可能です。**

外生変数の真価は：
- モデルの汎化性能向上
- 時系列構造の学習支援
- 購買行動等の副次的予測

にあり、数字的中率の大幅向上を約束するものではありません。

---

**作成者**: AI Research Assistant  
**最終更新**: 2025年12月14日  
**バージョン**: 1.0.0  
**ライセンス**: MIT


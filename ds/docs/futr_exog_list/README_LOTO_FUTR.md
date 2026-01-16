# 🎯 外生変数生成＆PostgreSQL格納 完全ガイド

## 📁 ファイル配置

```
/mnt/e/env/ts/tslib/
├── ds/
│   ├── bin/
│   │   └── run_loto_futr.sh          # 実行スクリプト
│   ├── src/
│   │   └── create_loto_futr_exog.py  # メインPythonスクリプト
│   ├── logs/
│   │   └── loto_futr_exog.log        # ログファイル
│   └── data/
│       └── loto_futr_backup.csv      # CSVバックアップ（オプション）
└── test.ipynb                         # Jupyter Notebook実行用
```

---

## 🚀 セットアップ

### 1. ファイル配置

```bash
# ディレクトリ移動
cd /mnt/e/env/ts/tslib

# Pythonスクリプト配置
cp ~/create_loto_futr_exog.py ds/src/

# 実行スクリプト配置
cp ~/run_loto_futr.sh ds/bin/

# 実行権限付与
chmod +x ds/bin/run_loto_futr.sh
```

### 2. 必要なPythonパッケージインストール

```bash
# 仮想環境アクティベート（必要に応じて）
source /path/to/your/venv/bin/activate

# パッケージインストール
pip install pandas numpy sqlalchemy psycopg2-binary jpholiday
```

### 3. PostgreSQL接続設定

`ds/src/create_loto_futr_exog.py` の接続情報を編集：

```python
DB_CONFIG = {
    'host': 'localhost',      # ← 必要に応じて変更
    'port': 5432,             # ← 必要に応じて変更
    'database': 'postgres',   # ← 必要に応じて変更
    'user': 'postgres',       # ← 必要に応じて変更
    'password': 'postgres'    # ← 必要に応じて変更
}
```

---

## 💻 実行方法

### 方法1: シェルスクリプト実行（推奨）

```bash
cd /mnt/e/env/ts/tslib
./ds/bin/run_loto_futr.sh
```

### 方法2: Pythonスクリプト直接実行

```bash
cd /mnt/e/env/ts/tslib/ds/src
python3 create_loto_futr_exog.py
```

### 方法3: Jupyter Notebook実行

1. `test.ipynb` を開く
2. 以下のセルを実行:

```python
import os
os.chdir('/mnt/e/env/ts/tslib/ds/src')

import create_loto_futr_exog
success = create_loto_futr_exog.main()
```

または

```python
!python3 /mnt/e/env/ts/tslib/ds/src/create_loto_futr_exog.py
```

---

## 📊 生成される外生変数

### カテゴリ別変数数

| カテゴリ | 変数数 | 例 |
|---------|--------|-----|
| 基本カレンダー | 26 | year, month, day, quarter, week, ... |
| 周期的エンコーディング | 14 | month_sin, month_cos, dayofweek_sin, ... |
| トレンド | 6 | trend_linear, trend_squared, trend_log, ... |
| フーリエ項（k=1～5） | 40 | fourier_year_sin_1, fourier_month_cos_3, ... |
| イベント・祝日 | 19 | is_holiday, is_golden_week, is_bonus_month, ... |
| Bingo5特化 | 8 | nth_wednesday_of_month, draw_number, ... |
| 周期変数（freq2～freq100） | 99 | freq2, freq3, ..., freq100 |

**総計**: **約210変数**

### 周期変数（freq2～freq100）について

| 変数 | 説明 | 値の例 |
|------|------|--------|
| freq2 | 2周期 | 1,2,1,2,1,2,... |
| freq3 | 3周期 | 1,2,3,1,2,3,... |
| freq7 | 7周期（週） | 1,2,3,4,5,6,7,1,2,... |
| freq12 | 12周期（月） | 1,2,3,...,12,1,2,... |
| freq52 | 52周期（年） | 1,2,3,...,52,1,2,... |
| freq100 | 100周期 | 1,2,3,...,100,1,2,... |

---

## 🗄️ データベーススキーマ

### テーブル: `public.loto_futr`

```sql
CREATE TABLE public.loto_futr (
    id SERIAL PRIMARY KEY,
    ds DATE NOT NULL UNIQUE,
    
    -- 基本カレンダー
    year INTEGER,
    month INTEGER,
    day INTEGER,
    quarter INTEGER,
    week INTEGER,
    dayofweek INTEGER,
    dayofyear INTEGER,
    week_of_month INTEGER,
    
    -- フラグ
    is_month_start INTEGER,
    is_month_end INTEGER,
    is_quarter_start INTEGER,
    is_quarter_end INTEGER,
    is_weekend INTEGER,
    is_holiday INTEGER,
    
    -- 周期的エンコーディング
    month_sin DOUBLE PRECISION,
    month_cos DOUBLE PRECISION,
    dayofweek_sin DOUBLE PRECISION,
    dayofweek_cos DOUBLE PRECISION,
    
    -- トレンド
    trend_linear INTEGER,
    trend_squared BIGINT,
    trend_log DOUBLE PRECISION,
    
    -- フーリエ項（動的追加）
    fourier_year_sin_1 DOUBLE PRECISION,
    fourier_year_cos_1 DOUBLE PRECISION,
    ...
    
    -- 周期変数（動的追加）
    freq2 INTEGER,
    freq3 INTEGER,
    ...
    freq100 INTEGER,
    
    -- Bingo5特化
    nth_wednesday_of_month INTEGER,
    draw_number INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_loto_futr_ds ON public.loto_futr(ds);
CREATE INDEX idx_loto_futr_year_month ON public.loto_futr(year, month);
```

### データ範囲

- **開始日**: 1994-10-01（最初の水曜日）
- **終了日**: 現在日付 + 10年
- **頻度**: 毎週水曜日
- **総レコード数**: 約1,600～1,800行

---

## 🔍 実行後の確認

### 1. データ件数確認

```sql
SELECT COUNT(*) FROM public.loto_futr;
```

### 2. 日付範囲確認

```sql
SELECT 
    MIN(ds) as 最小日付,
    MAX(ds) as 最大日付,
    COUNT(DISTINCT EXTRACT(YEAR FROM ds)) as 年数
FROM public.loto_futr;
```

### 3. サンプルデータ

```sql
SELECT 
    ds, year, month, day,
    month_sin, month_cos,
    trend_linear,
    is_holiday,
    nth_wednesday_of_month,
    freq2, freq7, freq52
FROM public.loto_futr
ORDER BY ds
LIMIT 10;
```

### 4. カラム一覧

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'loto_futr'
ORDER BY ordinal_position;
```

---

## 🔧 トラブルシューティング

### エラー1: PostgreSQL接続失敗

```
sqlalchemy.exc.OperationalError: could not connect to server
```

**解決策**:
1. PostgreSQLサービス起動確認
   ```bash
   sudo systemctl status postgresql
   sudo systemctl start postgresql  # 停止している場合
   ```

2. 接続情報確認
   ```python
   DB_CONFIG = {
       'host': 'localhost',  # ← 正しいホスト
       'port': 5432,         # ← 正しいポート
       ...
   }
   ```

### エラー2: 権限エラー

```
ERROR: permission denied for table loto_futr
```

**解決策**:
```sql
GRANT ALL PRIVILEGES ON TABLE public.loto_futr TO postgres;
GRANT USAGE, SELECT ON SEQUENCE loto_futr_id_seq TO postgres;
```

### エラー3: jpholidayインストールエラー

```
ModuleNotFoundError: No module named 'jpholiday'
```

**解決策**:
```bash
pip install jpholiday
```

### エラー4: メモリ不足

```
MemoryError: Unable to allocate array
```

**解決策**:
`create_loto_futr_exog.py` の `chunksize` を小さく:
```python
df.to_sql(
    ...
    chunksize=500  # ← 1000から500に変更
)
```

---

## 📈 使用例（NeuralForecast）

### 1. 外生変数読み込み

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# 未来の外生変数読み込み
futr_df = pd.read_sql("""
    SELECT 
        ds,
        month_sin, month_cos,
        dayofweek_sin, dayofweek_cos,
        dayofyear_sin, dayofyear_cos,
        trend_linear,
        is_holiday, is_weekend,
        nth_wednesday_of_month,
        freq52
    FROM public.loto_futr
    WHERE ds >= '2025-01-01'
    ORDER BY ds
""", engine)

print(futr_df.head())
```

### 2. NeuralForecastモデル訓練

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# 外生変数リスト
futr_exog_list = [
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'trend_linear',
    'is_holiday', 'is_weekend',
    'nth_wednesday_of_month',
    'freq52'
]

# モデル定義
models = [
    NHITS(
        h=12,
        input_size=52,
        futr_exog_list=futr_exog_list,
        scaler_type='robust',  # 必須！
        max_steps=1000
    )
]

# 訓練
nf = NeuralForecast(models=models, freq='W-WED')
nf.fit(df=df_train)

# 予測
predictions = nf.predict(futr_df=futr_df)
```

---

## 📝 ログ確認

```bash
# ログファイル確認
cat /mnt/e/env/ts/tslib/ds/logs/loto_futr_exog.log

# リアルタイムログ監視
tail -f /mnt/e/env/ts/tslib/ds/logs/loto_futr_exog.log
```

---

## 🎯 推奨外生変数セット

### Essential（15変数）- 初期検証用

```python
ESSENTIAL_FUTR_EXOG = [
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'trend_linear',
    'is_holiday', 'is_weekend',
    'nth_wednesday_of_month',
    'is_payday_25th',
    'is_golden_week',
    'is_new_year_holiday',
    'freq52'
]
```

### Standard（35変数）- 本番運用推奨

```python
STANDARD_FUTR_EXOG = ESSENTIAL_FUTR_EXOG + [
    'week_sin', 'week_cos',
    'quarter_sin', 'quarter_cos',
    'fourier_year_sin_1', 'fourier_year_cos_1',
    'fourier_year_sin_2', 'fourier_year_cos_2',
    'trend_squared', 'trend_log',
    'is_before_holiday', 'is_after_holiday',
    'is_month_end', 'is_quarter_end',
    'days_to_month_end',
    'is_obon',
    'is_bonus_month',
    'is_fiscal_year_end',
    'draw_number',
    'freq7', 'freq12'
]
```

---

## 📚 参考資料

- **NeuralForecast公式**: https://nixtlaverse.nixtla.io/neuralforecast/
- **外生変数ガイド**: `ds/docs/futr_exog_list/comprehensive_futr_exog_list.md`
- **実装ガイド**: `bingo5_futr_exog_implementation_guide.md`

---

## ✅ チェックリスト

- [ ] PostgreSQL起動確認
- [ ] Pythonパッケージインストール
- [ ] DB接続情報設定
- [ ] ファイル配置完了
- [ ] 実行権限付与
- [ ] スクリプト実行
- [ ] データ件数確認
- [ ] サンプルデータ確認
- [ ] ログ確認
- [ ] CSVバックアップ（オプション）

---

**作成日**: 2025年12月14日  
**バージョン**: 1.0.0  
**ライセンス**: MIT

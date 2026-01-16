# test.ipynb用のセル内容
## Jupyter Notebookで外生変数生成を実行

---

## 📋 Markdown Cell 1: タイトル

```markdown
# 🎯 外生変数生成＆PostgreSQL格納
## Bingo5予測用 1994-10-01 ～ 現在+10年

**目的**: public.loto_futrテーブルに外生変数を格納

**機能**:
- 基本カレンダー変数（30種）
- 周期的エンコーディング（14種）
- トレンド変数（6種）
- フーリエ項（k=1～5）
- イベント・祝日（20種）
- Bingo5特化変数（10種）
- 周期変数（freq2～freq100）

**総変数数**: 200+
```

---

## 💻 Code Cell 1: 環境確認

```python
# 環境確認
import sys
import pandas as pd
import numpy as np

print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

# 作業ディレクトリ
import os
os.chdir('/mnt/e/env/ts/tslib/ds/src')
print(f"作業ディレクトリ: {os.getcwd()}")

# ファイル確認
if os.path.exists('create_loto_futr_exog.py'):
    print("✓ create_loto_futr_exog.py 存在確認")
else:
    print("✗ create_loto_futr_exog.py が見つかりません")
```

---

## 📋 Markdown Cell 2: 実行前の準備

```markdown
## 実行前の準備

### PostgreSQL接続情報確認

スクリプト内のDB接続情報を確認してください：

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}
```

必要に応じて変更してください。
```

---

## 💻 Code Cell 2: メイン実行（方法1: importして実行）

```python
# スクリプトをimportして実行
import create_loto_futr_exog

# メイン関数実行
success = create_loto_futr_exog.main()

if success:
    print("\n" + "="*60)
    print("✓ 外生変数生成＆格納 完了！")
    print("="*60)
else:
    print("\n" + "="*60)
    print("✗ エラーが発生しました")
    print("="*60)
```

---

## 💻 Code Cell 3: メイン実行（方法2: シェルコマンド）

```python
# シェルコマンドで実行
!python3 /mnt/e/env/ts/tslib/ds/src/create_loto_futr_exog.py
```

---

## 💻 Code Cell 4: メイン実行（方法3: 実行スクリプト経由）

```python
# 実行スクリプト経由
!/bin/bash /mnt/e/env/ts/tslib/ds/bin/run_loto_futr.sh
```

---

## 📋 Markdown Cell 3: 実行後の確認

```markdown
## 実行後の確認

データベースに正しく格納されたか確認します。
```

---

## 💻 Code Cell 5: データ確認

```python
from sqlalchemy import create_engine, text
import pandas as pd

# PostgreSQL接続
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# 統計情報取得
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total_records,
            MIN(ds) as min_date,
            MAX(ds) as max_date,
            COUNT(DISTINCT EXTRACT(YEAR FROM ds)) as years,
            COUNT(DISTINCT EXTRACT(MONTH FROM ds)) as months
        FROM public.loto_futr
    """))
    stats = result.fetchone()

print("="*60)
print("📊 public.loto_futr テーブル統計")
print("="*60)
print(f"総レコード数: {stats[0]:,}")
print(f"最小日付: {stats[1]}")
print(f"最大日付: {stats[2]}")
print(f"年数: {stats[3]}")
print(f"月数: {stats[4]}")
print("="*60)
```

---

## 💻 Code Cell 6: サンプルデータ表示

```python
# サンプルデータ（最初の10行）
query = """
SELECT 
    ds, year, month, day, quarter,
    month_sin, month_cos,
    dayofweek_sin, dayofweek_cos,
    trend_linear,
    is_holiday, is_weekend,
    nth_wednesday_of_month,
    freq2, freq3, freq10, freq52, freq100
FROM public.loto_futr
ORDER BY ds
LIMIT 10
"""

df_sample = pd.read_sql(query, engine)
print("\n📋 サンプルデータ（最初の10行）:")
display(df_sample)
```

---

## 💻 Code Cell 7: カラム一覧確認

```python
# 全カラム一覧
query_columns = """
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'loto_futr'
ORDER BY ordinal_position
"""

df_columns = pd.read_sql(query_columns, engine)
print(f"\n📊 カラム一覧（{len(df_columns)}カラム）:")
display(df_columns)
```

---

## 💻 Code Cell 8: 周期変数確認

```python
# 周期変数（freq2～freq100）の確認
freq_cols = [f'freq{i}' for i in range(2, 101)]

# 存在確認
existing_freq = df_columns[df_columns['column_name'].str.startswith('freq')]['column_name'].tolist()
print(f"周期変数数: {len(existing_freq)}")

# サンプル表示
query_freq = f"""
SELECT ds, {', '.join(existing_freq[:10])}
FROM public.loto_futr
ORDER BY ds
LIMIT 20
"""

df_freq_sample = pd.read_sql(query_freq, engine)
print("\n📋 周期変数サンプル（freq2～freq11）:")
display(df_freq_sample)
```

---

## 💻 Code Cell 9: 統計サマリー

```python
# 数値カラムの統計
query_stats = """
SELECT 
    COUNT(*) as count,
    AVG(trend_linear) as avg_trend,
    MIN(month_sin) as min_month_sin,
    MAX(month_sin) as max_month_sin,
    SUM(is_holiday) as total_holidays,
    SUM(is_weekend) as total_weekends,
    AVG(year_progress) as avg_year_progress
FROM public.loto_futr
"""

df_stats = pd.read_sql(query_stats, engine)
print("\n📊 統計サマリー:")
display(df_stats.T)
```

---

## 💻 Code Cell 10: 年別レコード数

```python
# 年別レコード数
query_yearly = """
SELECT 
    year,
    COUNT(*) as records,
    MIN(ds) as first_date,
    MAX(ds) as last_date
FROM public.loto_futr
GROUP BY year
ORDER BY year
"""

df_yearly = pd.read_sql(query_yearly, engine)
print("\n📊 年別レコード数:")
display(df_yearly)

# グラフ表示
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.bar(df_yearly['year'], df_yearly['records'])
plt.xlabel('Year')
plt.ylabel('Records')
plt.title('年別レコード数')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 💻 Code Cell 11: 祝日分析

```python
# 祝日分析
query_holidays = """
SELECT 
    year,
    SUM(is_holiday) as holidays,
    SUM(is_golden_week) as golden_weeks,
    SUM(is_obon) as obons,
    SUM(is_new_year_holiday) as new_years
FROM public.loto_futr
GROUP BY year
ORDER BY year
"""

df_holidays = pd.read_sql(query_holidays, engine)
print("\n📊 年別祝日・イベント数:")
display(df_holidays.tail(10))
```

---

## 💻 Code Cell 12: 外生変数エクスポート（CSVバックアップ）

```python
# CSVにエクスポート（バックアップ）
query_all = "SELECT * FROM public.loto_futr ORDER BY ds"
df_all = pd.read_sql(query_all, engine)

# 保存
output_path = '/mnt/e/env/ts/tslib/ds/data/loto_futr_backup.csv'
df_all.to_csv(output_path, index=False)

print(f"✓ CSVバックアップ保存完了")
print(f"  パス: {output_path}")
print(f"  サイズ: {df_all.shape}")
print(f"  カラム数: {len(df_all.columns)}")
```

---

## 📋 Markdown Cell 4: 完了メッセージ

```markdown
## ✅ 完了

外生変数が正常に生成され、`public.loto_futr` テーブルに格納されました！

### 次のステップ

1. **NeuralForecastでの使用**:
   ```python
   from neuralforecast import NeuralForecast
   from neuralforecast.models import NHITS
   
   # 外生変数読み込み
   futr_df = pd.read_sql(
       "SELECT * FROM public.loto_futr WHERE ds >= '2024-01-01'",
       engine
   )
   ```

2. **特徴量選択**:
   - Essential: 15変数
   - Standard: 35変数
   - Advanced: 70+変数

3. **モデル訓練**:
   ```python
   models = [NHITS(h=12, futr_exog_list=[...], scaler_type='robust')]
   ```

### トラブルシューティング

- **接続エラー**: PostgreSQLサービス起動確認
- **権限エラー**: PostgreSQLユーザー権限確認
- **メモリエラー**: chunksizeを小さく（500など）
```

---

## 完了！

これらのセルを順番にJupyter Notebookで実行してください。

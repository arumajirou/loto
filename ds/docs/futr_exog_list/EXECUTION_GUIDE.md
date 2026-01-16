# 🚀 外生変数生成 完全実行ガイド
## 1994-10-01 ～ 現在+10年 のデータをPostgreSQLに格納

---

## 📦 提供ファイル

1. **create_loto_futr_exog.py** - メインPythonスクリプト
2. **run_loto_futr.sh** - 実行用シェルスクリプト
3. **test_ipynb_cells.md** - Jupyter Notebook用セル集
4. **README_LOTO_FUTR.md** - 詳細ドキュメント

---

## 🎯 Step 1: ファイル配置

```bash
# プロジェクトディレクトリに移動
cd /mnt/e/env/ts/tslib

# Pythonスクリプト配置
cp ~/create_loto_futr_exog.py ds/src/
# または
mv /mnt/user-data/outputs/create_loto_futr_exog.py ds/src/

# シェルスクリプト配置
cp ~/run_loto_futr.sh ds/bin/
# または
mv /mnt/user-data/outputs/run_loto_futr.sh ds/bin/

# 実行権限付与
chmod +x ds/bin/run_loto_futr.sh
chmod +x ds/src/create_loto_futr_exog.py
```

**確認**:
```bash
ls -lh ds/src/create_loto_futr_exog.py
ls -lh ds/bin/run_loto_futr.sh
```

---

## 🔧 Step 2: PostgreSQL設定確認

### 2-1. PostgreSQL起動確認

```bash
sudo systemctl status postgresql
```

起動していない場合:
```bash
sudo systemctl start postgresql
```

### 2-2. 接続情報確認

`ds/src/create_loto_futr_exog.py` を開いて接続情報を確認・編集:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'  # ← あなたのパスワード
}
```

### 2-3. 接続テスト

```bash
psql -h localhost -U postgres -d postgres -c "SELECT version();"
```

---

## 📚 Step 3: Pythonパッケージインストール

```bash
# 仮想環境アクティベート（必要に応じて）
conda activate nc
# または
source /path/to/venv/bin/activate

# パッケージインストール
pip install pandas numpy sqlalchemy psycopg2-binary jpholiday
```

**確認**:
```bash
python3 -c "import pandas, numpy, sqlalchemy, jpholiday; print('OK')"
```

---

## 🚀 Step 4: 実行

### 方法A: シェルスクリプト実行（推奨）

```bash
cd /mnt/e/env/ts/tslib
./ds/bin/run_loto_futr.sh
```

**期待される出力**:
```
========================================
外生変数生成＆PostgreSQL格納
========================================
[1/2] Python環境確認...
Python 3.x.x
[2/2] 外生変数生成実行...
2025-12-14 11:00:00 - INFO - ============================================================
2025-12-14 11:00:00 - INFO - 外生変数生成＆PostgreSQL格納 開始
...
2025-12-14 11:01:00 - INFO - データ挿入: 完了（1,600レコード）
========================================
✓ 正常終了
========================================
```

### 方法B: Python直接実行

```bash
cd /mnt/e/env/ts/tslib/ds/src
python3 create_loto_futr_exog.py
```

### 方法C: Jupyter Notebook実行

**`test.ipynb` で以下を実行**:

#### Cell 1: 環境確認

```python
import os
import sys

os.chdir('/mnt/e/env/ts/tslib/ds/src')
print(f"作業ディレクトリ: {os.getcwd()}")

if os.path.exists('create_loto_futr_exog.py'):
    print("✓ スクリプト存在確認")
else:
    print("✗ スクリプトが見つかりません")
```

#### Cell 2: メイン実行

```python
import create_loto_futr_exog

success = create_loto_futr_exog.main()

if success:
    print("\n✓ 完了！")
else:
    print("\n✗ エラー発生")
```

---

## ✅ Step 5: 確認

### 5-1. データ件数確認

```bash
psql -h localhost -U postgres -d postgres -c "SELECT COUNT(*) FROM public.loto_futr;"
```

**期待される結果**: 約1,600～1,800行

### 5-2. 日付範囲確認

```bash
psql -h localhost -U postgres -d postgres -c "
SELECT 
    MIN(ds) as 最小日付,
    MAX(ds) as 最大日付,
    COUNT(*) as 総件数
FROM public.loto_futr;
"
```

**期待される結果**:
```
   最小日付    |   最大日付    | 総件数
--------------+--------------+-------
 1994-10-05   | 2035-12-26   | 1,xxx
```

### 5-3. サンプルデータ確認

```bash
psql -h localhost -U postgres -d postgres -c "
SELECT 
    ds, year, month, 
    month_sin, month_cos,
    is_holiday, 
    nth_wednesday_of_month,
    freq2, freq52
FROM public.loto_futr
ORDER BY ds
LIMIT 5;
"
```

### 5-4. カラム数確認

```bash
psql -h localhost -U postgres -d postgres -c "
SELECT COUNT(*) as カラム数
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'loto_futr';
"
```

**期待される結果**: 約210カラム

### 5-5. ログ確認

```bash
cat ds/logs/loto_futr_exog.log
```

---

## 📊 Step 6: Jupyter Notebookで詳細確認

`test.ipynb` で以下を実行:

### Cell 1: 統計情報

```python
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total_records,
            MIN(ds) as min_date,
            MAX(ds) as max_date,
            COUNT(DISTINCT EXTRACT(YEAR FROM ds)) as years
        FROM public.loto_futr
    """))
    stats = result.fetchone()

print("="*60)
print("📊 統計情報")
print("="*60)
print(f"総レコード数: {stats[0]:,}")
print(f"最小日付: {stats[1]}")
print(f"最大日付: {stats[2]}")
print(f"年数: {stats[3]}")
print("="*60)
```

### Cell 2: サンプルデータ表示

```python
query = """
SELECT 
    ds, year, month, day,
    month_sin, month_cos,
    dayofweek_sin, dayofweek_cos,
    trend_linear,
    is_holiday, is_weekend,
    nth_wednesday_of_month,
    freq2, freq7, freq52, freq100
FROM public.loto_futr
ORDER BY ds
LIMIT 10
"""

df_sample = pd.read_sql(query, engine)
print("\n📋 サンプルデータ:")
display(df_sample)
```

### Cell 3: 周期変数確認

```python
# freq2の値確認（1,2,1,2,... となっているか）
query_freq2 = """
SELECT ds, freq2, freq3, freq7, freq12, freq52
FROM public.loto_futr
ORDER BY ds
LIMIT 20
"""

df_freq = pd.read_sql(query_freq2, engine)
print("\n📊 周期変数の値:")
display(df_freq)

# freq2が正しく1,2,1,2,... となっているか確認
print("\nfreq2のユニーク値:", sorted(df_freq['freq2'].unique()))
print("freq3のユニーク値:", sorted(df_freq['freq3'].unique()))
print("freq7のユニーク値:", sorted(df_freq['freq7'].unique()))
```

### Cell 4: 年別レコード数

```python
import matplotlib.pyplot as plt

query_yearly = """
SELECT 
    year,
    COUNT(*) as records
FROM public.loto_futr
GROUP BY year
ORDER BY year
"""

df_yearly = pd.read_sql(query_yearly, engine)

plt.figure(figsize=(15, 5))
plt.bar(df_yearly['year'], df_yearly['records'])
plt.xlabel('Year')
plt.ylabel('Records')
plt.title('年別レコード数')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n平均レコード数/年: {df_yearly['records'].mean():.1f}")
```

### Cell 5: カラム一覧

```python
query_columns = """
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'loto_futr'
ORDER BY ordinal_position
"""

df_columns = pd.read_sql(query_columns, engine)
print(f"\n📊 全カラム一覧（{len(df_columns)}カラム）:")
display(df_columns)
```

---

## 🎯 Step 7: NeuralForecastで使用

### Jupyter Notebookで実行:

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

# 外生変数読み込み（2025年以降）
futr_df_query = """
SELECT 
    'N1' as unique_id,
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
"""

futr_df = pd.read_sql(futr_df_query, engine)
print(f"未来データフレーム: {futr_df.shape}")
display(futr_df.head())

# futr_exog_list定義
futr_exog_list = [
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'trend_linear',
    'is_holiday', 'is_weekend',
    'nth_wednesday_of_month',
    'freq52'
]

print(f"\n外生変数リスト: {futr_exog_list}")
```

---

## 🔍 トラブルシューティング

### Q1. "No module named 'jpholiday'" エラー

**A1.**
```bash
pip install jpholiday
```

### Q2. PostgreSQL接続エラー

**A2.**
```bash
# PostgreSQL起動
sudo systemctl start postgresql

# 接続テスト
psql -h localhost -U postgres -d postgres -c "SELECT 1;"
```

### Q3. 実行権限エラー

**A3.**
```bash
chmod +x ds/bin/run_loto_futr.sh
chmod +x ds/src/create_loto_futr_exog.py
```

### Q4. メモリ不足

**A4.**  
`create_loto_futr_exog.py` の `chunksize` を500に変更:
```python
df.to_sql(..., chunksize=500)
```

### Q5. freq変数が正しく生成されない

**A5.**  
確認クエリ:
```sql
SELECT freq2, COUNT(*) 
FROM public.loto_futr 
GROUP BY freq2 
ORDER BY freq2;
```
期待される結果: freq2は1,2の2種類のみ

---

## 📋 実行コマンド早見表

```bash
# ファイル配置
cp /mnt/user-data/outputs/create_loto_futr_exog.py /mnt/e/env/ts/tslib/ds/src/
cp /mnt/user-data/outputs/run_loto_futr.sh /mnt/e/env/ts/tslib/ds/bin/
chmod +x /mnt/e/env/ts/tslib/ds/bin/run_loto_futr.sh

# 実行
cd /mnt/e/env/ts/tslib
./ds/bin/run_loto_futr.sh

# 確認
psql -h localhost -U postgres -d postgres -c "SELECT COUNT(*) FROM public.loto_futr;"

# ログ確認
cat ds/logs/loto_futr_exog.log
```

---

## ✅ 完了チェックリスト

- [ ] PostgreSQL起動確認
- [ ] Pythonパッケージインストール
- [ ] ファイル配置完了
- [ ] DB接続情報設定
- [ ] 実行権限付与
- [ ] **スクリプト実行**
- [ ] データ件数確認（1,600～1,800行）
- [ ] 日付範囲確認（1994-10-05 ～ 2035年頃）
- [ ] カラム数確認（約210カラム）
- [ ] freq変数確認（freq2=1,2、freq3=1,2,3など）
- [ ] サンプルデータ確認
- [ ] ログ確認

---

## 🎉 完了！

これで `public.loto_futr` テーブルに**約210種類**の外生変数が格納されました！

### 次のステップ

1. **NeuralForecastで使用** (上記 Step 7 参照)
2. **特徴量選択** (Essential/Standard/Advanced)
3. **モデル訓練・予測**

---

**作成日**: 2025年12月14日  
**バージョン**: 1.0.0

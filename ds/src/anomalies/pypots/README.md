# PyPOTS 異常検知 (DB → Windowing → Fit/Detect → DB)

このフォルダは **縦持ち** (loto, unique_id, ts_type, ds, y) の時系列を、PyPOTS の入力仕様
`X: [n_samples, n_steps, n_features]` に合わせて **ウィンドウ化**し、
学習→推論して結果を `anomaly.public.pypots` に保存するための実行一式です。

## 実行

```bash
(ts) az@az:/mnt/e/env/ts/tslib/ds/src/anomalies/pypots$ python anomaly_pypots.py --config ./config.ini
```

GPU を使う例:

```bash
python anomaly_pypots.py --config ./config.ini --device cuda:0
```

## カラム命名

特徴量は衝突回避のため、必ず `hist_pypots__{pipeline}__{field}` 形式で出力します。

## DB 接続

`config.ini` で `db_source.name=dataset` を指定すると、環境変数 `DB_URL_DATASET` を見に行きます。
（`postgresql+psycopg2://...` 形式など）

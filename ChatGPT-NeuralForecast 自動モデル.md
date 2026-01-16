# NeuralForecast 自動モデル

**User:** Anonymous (tajimaharu98@gmail.com)  
**Created:** 1/15/2026 7:02:28  
**Updated:** 1/15/2026 7:50:09  
**Exported:** 1/15/2026 7:57:58  
**Link:** [https://chatgpt.com/c/696811f0-b1f0-8324-bc4b-8097d4920985](https://chatgpt.com/c/696811f0-b1f0-8324-bc4b-8097d4920985)  

## Response:
思考時間: 16s

結論：`/mnt/e/env/ts/tslib/neuralforecast.ipynb` を「**実装→即テスト→即DB確認**」の司令塔にして、実体は `/mnt/e/env/ts/tslib/src/` に小さく切り出し、`pytest` は **単体（DataFrame）→統合（DB）→回帰（過去run再現）** の順に積むのが業務的に最短で堅いです。

* * *

業務レベル詳細 実装計画表（Notebook駆動＋pytest並走）
----------------------------------

> 列の見方：  
> **NBセル**＝Notebook内のセル群（見出し単位で区切る想定）  
> **実装先**＝最終的に切り出すPythonファイル（フルパス）  
> **pytest**＝追加するテスト（フルパス）  
> **Done**＝受入条件（動作確認の定義）

| ID | 作業項目（細分化） | NBセル（見出し） | 実装先（フルパス） | pytest（フルパス） | Done（受入条件） |
| --- | --- | --- | --- | --- | --- |
| 001 | Notebookの実行基盤（再実行・自動反映） | 0\. Notebook Setup | `/mnt/e/env/ts/tslib/src/nf_app/__init__.py` | ー | Notebook再実行で副作用が残らない（セル順で再現） |
| 002 | 設定管理（DB URL/Artifact Root/RunMode） | 0\. Config | `/mnt/e/env/ts/tslib/src/nf_app/config.py` | `/mnt/e/env/ts/tslib/tests/test_config.py` | `.env`/環境変数で切替できる |
| 003 | ロギング（run\_id付き） | 0\. Logging | `/mnt/e/env/ts/tslib/src/nf_app/logging.py` | `/mnt/e/env/ts/tslib/tests/test_logging.py` | run\_idが全ログに入る |
| 004 | 例外ポリシー（失敗分類：Data/DB/Model） | 0\. Errors | `/mnt/e/env/ts/tslib/src/nf_app/errors.py` | `/mnt/e/env/ts/tslib/tests/test_errors.py` | 例外が分類されDBに保存可能 |

### A. DBスキーマ運用（model DB / dataset DB）

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 010 | model DB接続（SQLAlchemy/psycopg） | 1\. DB Connect | `/mnt/e/env/ts/tslib/src/nf_app/db.py` | `/mnt/e/env/ts/tslib/tests/test_db_smoke.py` | model/dataset両方に接続できる |
| 011 | DDL実行（初回セットアップ関数） | 1\. DB Migrate | `/mnt/e/env/ts/tslib/src/nf_app/migrations.py` | `/mnt/e/env/ts/tslib/tests/test_migrations.py` | 必須テーブルが作成される |
| 012 | model: feature\_prefix\_rules作成 & 初期投入 | 1\. DB Migrate | `migrations.py` | `test_migrations.py` | hist\_/futr\_/stat\_が登録済み |
| 013 | model: join\_rules作成 | 1\. DB Migrate | `migrations.py` | `test_migrations.py` | join\_rulesにINSERTできる |
| 014 | model: experiments/run\_plans/runs/model\_versions | 1\. DB Migrate | `migrations.py` | `test_migrations.py` | enabled planだけ抽出できる |
| 015 | dataset: forecast保存テーブル（preds/backtest） | 1\. DB Migrate | `migrations.py` | `test_migrations.py` | 予測がINSERTできる |

### B. dataset結合→内省→外生変数自動分類（拡張可能）

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 020 | information\_schema内省（列一覧・型・NULL） | 2\. Introspection | `/mnt/e/env/ts/tslib/src/nf_app/introspect.py` | `/mnt/e/env/ts/tslib/tests/test_introspect.py` | 列増減で壊れない |
| 021 | join\_rulesに基づくJOIN SQL生成（衝突回避） | 2\. Join SQL | `/mnt/e/env/ts/tslib/src/nf_app/query_builder.py` | `/mnt/e/env/ts/tslib/tests/test_query_builder.py` | JOIN SQLが安定生成される |
| 022 | dataset結合結果の取得（pandas read\_sql） | 2\. Load Joined | `/mnt/e/env/ts/tslib/src/nf_app/dataset_loader.py` | `/mnt/e/env/ts/tslib/tests/test_dataset_loader_unit.py` | DataFrameが返る（最低 unique\_id/ds/y） |
| 023 | prefix\_rules取得（DB→メモリ辞書） | 2\. Prefix Rules | `/mnt/e/env/ts/tslib/src/nf_app/feature_rules.py` | `/mnt/e/env/ts/tslib/tests/test_feature_rules.py` | ルール変更が即反映される |
| 024 | 列を役割に分類（hist/futr/stat/meta/ignore） | 3\. Feature Routing | `/mnt/e/env/ts/tslib/src/nf_app/feature_router.py` | `/mnt/e/env/ts/tslib/tests/test_feature_router.py` | hist\_/futr\_/stat\_が正しく分類 |
| 025 | `df`生成（unique\_id/ds/y＋hist外生） | 3\. Build df | `/mnt/e/env/ts/tslib/src/nf_app/builders/df_builder.py` | `/mnt/e/env/ts/tslib/tests/test_df_builder.py` | df列が仕様通り |
| 026 | `futr_df`生成（unique\_id/ds＋futr外生） | 3\. Build futr\_df | `/mnt/e/env/ts/tslib/src/nf_app/builders/futr_builder.py` | `/mnt/e/env/ts/tslib/tests/test_futr_builder.py` | 予測ホライズン分を満たす |
| 027 | `static_df`生成（stat\_をunique\_idで集約） | 3\. Build static\_df | `/mnt/e/env/ts/tslib/src/nf_app/builders/static_builder.py` | `/mnt/e/env/ts/tslib/tests/test_static_builder.py` | 1 unique\_id=1行になる |
| 028 | データ型整形（ds→datetime、y→float等） | 3\. Dtypes | `/mnt/e/env/ts/tslib/src/nf_app/transforms.py` | `/mnt/e/env/ts/tslib/tests/test_transforms.py` | NeuralForecast投入で型エラーなし |
| 029 | 欠損戦略（drop/impute/flag）をRunPlan化 | 3\. Missing Policy | `/mnt/e/env/ts/tslib/src/nf_app/missing.py` | `/mnt/e/env/ts/tslib/tests/test_missing.py` | ポリシーを変えても再現 |

### C. 品質ゲート（リーク・一貫性・再現性）

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 030 | 基本品質チェック（重複・欠損・時系列順） | 4\. Data Quality | `/mnt/e/env/ts/tslib/src/nf_app/quality/basic_checks.py` | `/mnt/e/env/ts/tslib/tests/test_basic_checks.py` | NGで例外が出る |
| 031 | 未来リーク検査（futr\_混入/未来参照） | 4\. Leak Audit | `/mnt/e/env/ts/tslib/src/nf_app/quality/leak_audit.py` | `/mnt/e/env/ts/tslib/tests/test_leak_audit.py` | リーク疑いをブロック |
| 032 | データ署名（期間/件数/列/ハッシュ） | 4\. Data Signature | `/mnt/e/env/ts/tslib/src/nf_app/repro/signature.py` | `/mnt/e/env/ts/tslib/tests/test_signature.py` | 同条件で署名が一致 |
| 033 | “事前検査（preflight）”統合 | 4\. Preflight | `/mnt/e/env/ts/tslib/src/nf_app/preflight.py` | `/mnt/e/env/ts/tslib/tests/test_preflight.py` | 実行前に落とせる |

### D. 実行計画（run\_plans）駆動：作る/再学習/予測/ドライ

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 040 | run\_plans読込（enabledのみ、優先度順） | 5\. Plan Loader | `/mnt/e/env/ts/tslib/src/nf_app/plans.py` | `/mnt/e/env/ts/tslib/tests/test_plans.py` | 実行対象が確定する |
| 041 | runsの状態遷移（PENDING→RUNNING→…） | 5\. Run State | `/mnt/e/env/ts/tslib/src/nf_app/runs.py` | `/mnt/e/env/ts/tslib/tests/test_runs.py` | 失敗時もFAILEDで残る |
| 042 | Dry-run厳密化（DB書込なし/限定） | 5\. Dry Mode | `/mnt/e/env/ts/tslib/src/nf_app/run_modes.py` | `/mnt/e/env/ts/tslib/tests/test_run_modes.py` | dryでDBが汚れない |
| 043 | モデルファクトリ（model\_name→クラス生成） | 6\. Model Factory | `/mnt/e/env/ts/tslib/src/nf_app/model_factory.py` | `/mnt/e/env/ts/tslib/tests/test_model_factory.py` | 指定名で生成できる |
| 044 | 既存モデル復元（artifact\_path→load） | 6\. Model Load | `/mnt/e/env/ts/tslib/src/nf_app/model_io.py` | `/mnt/e/env/ts/tslib/tests/test_model_io.py` | 予測のみが動く |
| 045 | 学習（fit）→予測（predict）パイプライン | 6\. Train & Predict | `/mnt/e/env/ts/tslib/src/nf_app/model_runner.py` | `/mnt/e/env/ts/tslib/tests/test_model_runner_unit.py` | 最小1モデルで完走 |
| 046 | futr\_df/static\_dfを渡す分岐 | 6\. Exog Wiring | `model_runner.py` | `test_model_runner_unit.py` | futr/statありで動く |
| 047 | save（成果物規約：run\_id/版） | 6\. Save | `model_io.py` | `test_model_io.py` | 保存先が一意で衝突しない |
| 048 | model\_versions登録（exogリスト二重化） | 6\. Registry Write | `/mnt/e/env/ts/tslib/src/nf_app/registry.py` | `/mnt/e/env/ts/tslib/tests/test_registry.py` | DBのexogと実体が一致 |

### E. Cross Validation / 評価指標 / 検定 / 可視化（Verification強化）

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 060 | CVテンプレ（n\_windows/step/refit） | 7\. CV | `/mnt/e/env/ts/tslib/src/nf_app/eval/cv.py` | `/mnt/e/env/ts/tslib/tests/test_cv.py` | backtestが出る |
| 061 | CV制約チェック（モデル種別ごと） | 7\. CV Constraints | `/mnt/e/env/ts/tslib/src/nf_app/eval/constraints.py` | `/mnt/e/env/ts/tslib/tests/test_constraints.py` | 実行前に弾ける |
| 062 | 指標（MAE/RMSE/sMAPE等） | 8\. Metrics | `/mnt/e/env/ts/tslib/src/nf_app/eval/metrics.py` | `/mnt/e/env/ts/tslib/tests/test_metrics.py` | 指標が保存可能形式で出る |
| 063 | 区間/分位点指標（coverage/pinball等） | 8\. Prob Metrics | `metrics.py` | `test_metrics.py` | 分位点ありモデルで計算 |
| 064 | 残差解析（自己相関/異常） | 9\. Diagnostics | `/mnt/e/env/ts/tslib/src/nf_app/eval/diagnostics.py` | `/mnt/e/env/ts/tslib/tests/test_diagnostics.py` | 検定統計量が出る |
| 065 | 予測差検定（DM等） | 9\. Statistical Tests | `/mnt/e/env/ts/tslib/src/nf_app/eval/tests.py` | `/mnt/e/env/ts/tslib/tests/test_stat_tests.py` | run間比較が可能 |
| 066 | datasetへの保存（preds/backtest） | 10\. Persist preds | `/mnt/e/env/ts/tslib/src/nf_app/persist/preds.py` | `/mnt/e/env/ts/tslib/tests/test_persist_preds.py` | DBに予測が入る |
| 067 | modelへの保存（metrics/stat\_tests） | 10\. Persist metrics | `/mnt/e/env/ts/tslib/src/nf_app/persist/metrics.py` | `/mnt/e/env/ts/tslib/tests/test_persist_metrics.py` | DBに評価が入る |
| 068 | 標準プロット（予測vs実測/誤差/区間） | 11\. Plots | `/mnt/e/env/ts/tslib/src/nf_app/viz/plots.py` | `/mnt/e/env/ts/tslib/tests/test_plots_smoke.py` | 画像が生成される |
| 069 | レポート生成（runごとHTML/PNGまとめ） | 11\. Report | `/mnt/e/env/ts/tslib/src/nf_app/viz/report.py` | `/mnt/e/env/ts/tslib/tests/test_report.py` | 1 run=1レポが出る |

### F. Autoモデル（HPO）・試行履歴・互換性

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 080 | HPO設定（backend/tune/optuna）をPlan化 | 12\. HPO | `/mnt/e/env/ts/tslib/src/nf_app/hpo/config.py` | `/mnt/e/env/ts/tslib/tests/test_hpo_config.py` | Planだけで切替可能 |
| 081 | trial履歴保存（hpo\_trials） | 12\. HPO Trials | `/mnt/e/env/ts/tslib/src/nf_app/hpo/trials.py` | `/mnt/e/env/ts/tslib/tests/test_hpo_trials.py` | 試行の完全ログが残る |
| 082 | 互換性チェック（モデル×backend） | 12\. HPO Compat | `/mnt/e/env/ts/tslib/src/nf_app/hpo/compat.py` | `/mnt/e/env/ts/tslib/tests/test_hpo_compat.py` | 不適合を事前に停止 |
| 083 | HPO結果→ベースモデル化→保存 | 12\. HPO Commit | `/mnt/e/env/ts/tslib/src/nf_app/hpo/commit.py` | `/mnt/e/env/ts/tslib/tests/test_hpo_commit.py` | 最良モデルが台帳登録 |

### G. 運用（監視/ドリフト/アラート）※後半フェーズ

| ID | 作業項目 | NBセル | 実装先 | pytest | Done |
| --- | --- | --- | --- | --- | --- |
| 100 | drift指標（入力/予測/実測） | 13\. Drift | `/mnt/e/env/ts/tslib/src/nf_app/ops/drift.py` | `/mnt/e/env/ts/tslib/tests/test_drift.py` | 日次集計が可能 |
| 101 | SLO/アラートルール（DB駆動） | 13\. Alerts | `/mnt/e/env/ts/tslib/src/nf_app/ops/alerts.py` | `/mnt/e/env/ts/tslib/tests/test_alerts.py` | 閾値超えが記録される |
| 102 | 監査ログ（誰が何を実行） | 13\. Audit | `/mnt/e/env/ts/tslib/src/nf_app/ops/audit.py` | `/mnt/e/env/ts/tslib/tests/test_audit.py` | 操作履歴が追える |

* * *

pytest運用の「業務っぽい」分割（破綻しないための型）
-----------------------------

*   **Unit（最初に増やす）**：DataFrameだけで完結（外生分類・builder・品質ゲート）  
    → `/mnt/e/env/ts/tslib/tests/test_feature_router.py` 等
*   **Integration（次に増やす）**：ローカルPostgreSQLに接続して、DDL→INSERT→SELECTを確認  
    → `/mnt/e/env/ts/tslib/tests/test_db_smoke.py`, `test_migrations.py`
*   **Regression（最後に増やす）**：過去run（artifact + model\_versions）を読み込み、同じ予測が出ること  
    → `/mnt/e/env/ts/tslib/tests/test_model_io.py`

* * *

Notebook（`/mnt/e/env/ts/tslib/neuralforecast.ipynb`）の章立てテンプレ
------------------------------------------------------------

1.  Setup（autoreload / config / logger）
2.  DB Connect & Migrate（model/dataset）
3.  Load Joined Data（join\_rules → df抽出）
4.  Feature Routing（prefix\_rules）
5.  Build df/futr\_df/static\_df
6.  Preflight（品質＋リーク＋署名）
7.  RunPlan Execute（dry→本番）
8.  Train/Predict/Save/Registry
9.  CV & Evaluation（metrics/tests）
10.  Visualization & Report
11.  HPO（必要なら）
12.  Ops（drift/alerts：後半）

* * *

この計画表どおりに進めると、Notebookで確かめながらも「最終的にコードが散らからず、DB駆動で拡張でき、pytestで守られる」状態になります。次は **ID 010〜029（DB→結合→分類→df生成）** をNotebookの最初の実装対象にして、セルとモジュールの雛形を切り出すのが最短ルートです。

![Create Google Docs](chrome-extension://iapioliapockkkikccgbiaalfhoieano/assets/g-docs.svg)

Export response as a Google Docs file



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
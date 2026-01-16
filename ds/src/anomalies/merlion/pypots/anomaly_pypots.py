# /mnt/e/env/ts/tslib/ds/src/anomalies/pypots/anomaly_pypots.py
import argparse
import configparser
import pandas as pd
from datetime import datetime
import uuid
import sys

from naming import get_output_columns
from pypots_runner import PyPOTSModelRunner
from db_utils import DBManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pipeline', default='default')
    args = parser.parse_args()

    # config読込
    config = configparser.ConfigParser()
    config.read(args.config)
    p_conf = dict(config['pypots'])
    p_conf['pipeline'] = args.pipeline

    db = DBManager()
    run_id = str(uuid.uuid4())[:8]
    created_ts = datetime.now()

    # データ取得
    query = f"SELECT * FROM {config['db_source']['table']} ORDER BY loto, unique_id, ts_type, ds"
    df = db.read_query(query)

    output_list = []
    cols = get_output_columns(args.pipeline)

    # 系列ごとにループ処理（デバッグのため並列化せず実行）
    for keys, group in df.groupby(['loto', 'unique_id', 'ts_type']):
        runner = PyPOTSModelRunner(p_conf)
        scores, is_anom = runner.run_inference(group)
        
        if scores is not None:
            group[cols['anomaly_score']] = scores
            group[cols['is_anomaly']] = is_anom
            group['run_id'] = run_id
            group['created_ts'] = created_ts
            group['model'] = p_conf['model']
            output_list.append(group)

    if output_list:
        final_df = pd.concat(output_list)
        target = config['db_target']['table'].split('.')
        db.write_table(final_df, target[-1], target[1])
        print(f"Successfully saved {len(final_df)} rows.")

if __name__ == "__main__":
    main()
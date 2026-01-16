# ds/src/base/loto_transformer.py

from typing import Dict, List
import pandas as pd
from loto_common import logger

class LotoTransformer:
    """時系列特徴量と横持ち特徴量を生成し、縦持ち形式に変換するクラス"""

    def __init__(self):
        # 新しい加工ロジックを追加する場合は、ここに追加する
        self.TRANSFORMS: Dict[str, str] = {
            "raw": "raw",                 # オリジナルy
            "cumsum": "cumsum",           # 累積和
            "roll3_sum": "rolling(3).sum()",    # 3期間合計
            "roll7_mean": "rolling(7).mean()",  # 7期間平均
            "diff1": "diff(1)",           # 1階差分
            "is_odd": "is_odd"            # 奇数フラグ
        }
        logger.info(f"定義された時系列変換: {list(self.TRANSFORMS.keys())}")
        
    def _apply_ts_transform(self, y_base: pd.Series, logic: str) -> pd.Series:
        """単一の時系列変換を適用する"""
        try:
            if logic == "raw":
                return y_base
            elif logic == "cumsum":
                return y_base.cumsum()
            elif logic.startswith("rolling"):
                parts = logic.split('.')
                window_str = parts[0].replace('rolling(', '').replace(')', '')
                window = int(window_str)
                agg_func = parts[1].lower().replace('()', '')
                roller = y_base.rolling(window=window, min_periods=1)
                
                if agg_func == 'sum':
                    return roller.sum()
                elif agg_func == 'mean':
                    return roller.mean()
                else:
                    logger.warning(f"不明な集計関数: {agg_func}")
                    return pd.Series(dtype='float64')
            elif logic.startswith("diff"):
                lag = int(logic.replace('diff(', '').replace(')', ''))
                return y_base.diff(lag)
            elif logic == "is_odd":
                return y_base % 2
            else:
                return pd.Series(dtype='float64')
        except Exception as e:
            logger.error(f"時系列変換エラー (Logic: {logic}): {e}")
            return pd.Series(dtype='float64')

    def transform_and_melt(self, df_long_base: pd.DataFrame) -> pd.DataFrame:
        """
        グループごとに時系列変換を適用し、横持ち特徴量と共に結果を縦持ち (Long Format) に変換
        """
        results_list: List[pd.DataFrame] = []
        # 'y_base' は LotoExtractor から来る。Float型に変換し、欠損値はNaNで保持。
        df_long_base['y_base'] = pd.to_numeric(df_long_base['y_base'], errors='coerce').astype('float64')
        
        # 'loto' と 'unique_id' で時系列グループ化
        groups = df_long_base.groupby(['loto', 'unique_id'], sort=False)

        # 'hist_' で始まる列を特定 (横持ち特徴量)
        hist_cols = [c for c in df_long_base.columns if c.startswith('hist_')]
        
        # 共通のID列
        id_cols = ['loto', 'unique_id', 'ds']

        for (loto, uid), group in groups:
            group = group.sort_values('ds').copy()
            y_base = group['y_base']
            
            # 1. 時系列変換の適用
            transformed_data: Dict[str, pd.Series] = {}
            for ts_type, logic in self.TRANSFORMS.items():
                transformed_data[ts_type] = self._apply_ts_transform(y_base, logic)
            
            # 2. 変換済みデータを新しい列として追加
            data_to_combine = {
                'loto': loto, 
                'unique_id': uid, 
                'ds': group['ds'], 
            }
            # hist_cols (横持ち特徴量) も含める
            data_to_combine.update({c: group[c] for c in hist_cols})
            data_to_combine.update(transformed_data)

            # 3. 全てのデータを含むDataFrameを作成
            df_to_melt = pd.DataFrame(data_to_combine)

            # 4. 縦持ち (Melt) に変換
            # value_name: 'y' (要件に合わせてy_transformedから変更)
            df_melt = pd.melt(
                df_to_melt,
                id_vars=id_cols + hist_cols, # ID列と横持ち特徴量
                value_vars=list(transformed_data.keys()),
                var_name='ts_type',
                value_name='y' # y_transformedをyに変更
            )
            
            results_list.append(df_melt)

        return pd.concat(results_list, ignore_index=True)
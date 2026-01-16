# ds/src/base/loto_extractor.py

import re
from typing import Dict, List, Tuple
import pandas as pd
import requests
from pathlib import Path

from loto_common import URLS, ENCODINGS_TO_TRY

class LotoExtractor:
    """ロトデータCSVを取得し、整形・縦持ち化するクラス"""

    def __init__(self):
        pass

    def _download_csv_bytes(self, url: str) -> bytes:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://loto-life.net/csv/download",
        }
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.content

    def _read_loto_csv(self, url: str) -> pd.DataFrame:
        raw = self._download_csv_bytes(url)
        last_err = None
        for enc in ENCODINGS_TO_TRY:
            try:
                return pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
            except Exception as e:
                last_err = e
        # 最終手段としてエラー置換で読み込みを試みる
        return pd.read_csv(pd.io.common.BytesIO(raw), encoding="cp932", encoding_errors="replace")

    def _to_int_series(self, s: pd.Series) -> pd.Series:
        """文字列を数値に変換（カンマ除去、数字抽出、欠損値はInt64のNaN）"""
        s2 = s.astype(str)
        s2 = s2.str.replace(",", "", regex=False)
        s2 = s2.str.extract(r"([0-9]+)", expand=False)
        return pd.to_numeric(s2, errors="coerce").astype("Int64")

    def _normalize_loto_df(self, df: pd.DataFrame, loto: str) -> pd.DataFrame:
        """データフレームの列名を標準化し、型を変換する"""
        df = df.copy()

        # 開催日 (ds) の処理
        if "開催日" in df.columns:
            df = df.rename(columns={"開催日": "ds"})
        if "ds" not in df.columns:
            raise RuntimeError(f"[{loto}] 開催日（ds）列が見つかりません: cols={list(df.columns)}")
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.date
        df = df[df["ds"].notna()]
        
        # Numbers3/4の抽選数字の処理
        if loto in ("numbers3", "numbers4") and "抽選数字" in df.columns:
            n = 3 if loto == "numbers3" else 4
            digits = df["抽選数字"].astype(str).str.extract(r"([0-9]+)", expand=False).fillna("")
            digits = digits.str[-n:].str.zfill(n)
            for i in range(1, n + 1):
                df[f"N{i}"] = self._to_int_series(digits.str.slice(i - 1, i))
            df = df.drop(columns=["抽選数字"])

        # 列名マッピング
        rename_map = {}
        for c in df.columns:
            m_num = re.fullmatch(r"第(\d+)数字", str(c))
            m_bn = re.fullmatch(r"ボーナス数字(\d+)", str(c))
            m_prize_n = re.fullmatch(r"(\d+)等口数", str(c))
            m_prize_m = re.fullmatch(r"(\d+)等賞金", str(c))

            if m_num: rename_map[c] = f"N{int(m_num.group(1))}"
            elif c == "ボーナス数字": rename_map[c] = "B1"
            elif m_bn: rename_map[c] = f"B{int(m_bn.group(1))}"
            elif m_prize_n: rename_map[c] = f"PN{int(m_prize_n.group(1))}"
            elif m_prize_m: rename_map[c] = f"PM{int(m_prize_m.group(1))}"
        
        abbrev = {
            "キャリーオーバー": "CO", "キャリーオーバー額": "CO",
            "ストレート口数": "STC", "ストレート賞金": "STM",
            "ボックス口数": "BXC", "ボックス賞金": "BXM",
            "セット(ストレート)口数": "SSC", "セット(ストレート)賞金": "SSM",
            "セット(ボックス)口数": "SBC", "セット(ボックス)賞金": "SBM",
            "ミニ口数": "MNC", "ミニ賞金": "MNM",
            "セット（ストレート）口数": "SSC", "セット（ストレート）賞金": "SSM",
            "セット（ボックス）口数": "SBC", "セット（ボックス）賞金": "SBM",
        }
        rename_map.update({k: v for k, v in abbrev.items() if k in df.columns})

        df = df.rename(columns=rename_map)

        # 数値列の型変換
        num_pat = re.compile(r"^(N\d+|PN\d+|PM\d+|B\d+|CO|STC|STM|BXC|BXM|SSC|SSM|SBC|SBM|MNC|MNM)$")
        for c in [c for c in df.columns if num_pat.match(str(c))]:
            df[c] = self._to_int_series(df[c])

        if "開催回" in df.columns:
            df = df.drop(columns=["開催回"])

        return df

    def _wide_to_long(self, df: pd.DataFrame, loto: str) -> pd.DataFrame:
        """整形済みデータフレームを縦持ち形式に変換する (loto_base相当)"""
        df = df.copy()
        
        # N1, N2, ... 列を特定
        n_cols = sorted([c for c in df.columns if re.fullmatch(r"N\d+", str(c))], key=lambda x: int(str(x)[1:]))
        if not n_cols:
            raise RuntimeError(f"[{loto}] N列が見つかりません: cols={list(df.columns)}")

        # melt (縦持ち化)
        id_vars = [c for c in df.columns if c not in n_cols]
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=n_cols,
            var_name="unique_id",
            value_name="y_base", # 一時的な名称
        )
        long_df["loto"] = loto
        long_df["unique_id"] = long_df["unique_id"].astype(str).str.upper()
        long_df["y_base"] = self._to_int_series(long_df["y_base"]).fillna(0).astype("Int64") # 一時的な名称

        # loto_baseの主要列の順番
        base_cols = ["loto", "ds", "unique_id", "y_base"] 
        rest_cols = [c for c in long_df.columns if c not in base_cols]
        return long_df[base_cols + rest_cols].copy()

    def extract_and_transform(self, lotos: List[str]) -> pd.DataFrame:
        """全ロトデータを取得・整形・縦持ち化し、一つのデータフレームに統合する"""
        long_list: List[pd.DataFrame] = []
        for loto in lotos:
            if loto not in URLS:
                raise RuntimeError(f"未知の loto: {loto}")
                
            print(f"Processing {loto}...")
            df_raw = self._read_loto_csv(URLS[loto])
            df_norm = self._normalize_loto_df(df_raw, loto=loto)
            df_long = self._wide_to_long(df_norm, loto=loto)
            long_list.append(df_long)

        df_all = pd.concat(long_list, ignore_index=True)
        # 列名を小文字に統一し、'hist_' prefixを付与
        rename_map = {}
        for c in df_all.columns:
            c_lower = c.lower()
            if c_lower not in ['loto', 'ds', 'unique_id', 'y_base']:
                rename_map[c] = f'hist_{c_lower}'
            else:
                rename_map[c] = c_lower
        
        df_all.rename(columns=rename_map, inplace=True)
        
        return df_all
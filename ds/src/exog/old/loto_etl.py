#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto-life CSV → 整形 → PostgreSQL（loto_base / loto_hist）へ保存
重複削除（Dedup）機能付き統合版

修正点: ON CONFLICT句が動作するように UNIQUE INDEX を明示的に作成するように変更

実行例:
  python loto_etl.py --password z --mode append
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import requests

# --------------------------
# 設定・定数
# --------------------------
URLS: Dict[str, str] = {
    "mini": "https://loto-life.net/csv/mini",
    "loto6": "https://loto-life.net/csv/loto6",
    "loto7": "https://loto-life.net/csv/loto7",
    "bingo5": "https://loto-life.net/csv/bingo5",
    "numbers3": "https://loto-life.net/csv/numbers3",
    "numbers4": "https://loto-life.net/csv/numbers4",
}

BASE_COLS = ["loto", "ds", "unique_id", "y"]
KEY_COLS = ["loto", "ds", "unique_id"]
ENCODINGS_TO_TRY = ["utf-8-sig", "cp932", "shift_jis", "utf-8"]


# --------------------------
# CSV取得 & 変換（pandas）
# --------------------------
def download_csv_bytes(url: str) -> bytes:
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


def read_loto_csv(url: str) -> pd.DataFrame:
    raw = download_csv_bytes(url)
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    return pd.read_csv(pd.io.common.BytesIO(raw), encoding="cp932", encoding_errors="replace")


def _to_int_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str)
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.extract(r"([0-9]+)", expand=False)
    return pd.to_numeric(s2, errors="coerce").astype("Int64")


def normalize_loto_df(df: pd.DataFrame, loto: str) -> pd.DataFrame:
    df = df.copy()

    if "開催日" in df.columns:
        df = df.rename(columns={"開催日": "ds"})
    if "ds" not in df.columns:
        raise RuntimeError(f"[{loto}] 開催日（ds）列が見つかりません: cols={list(df.columns)}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.date
    df = df[df["ds"].notna()]

    if loto in ("numbers3", "numbers4") and "抽選数字" in df.columns:
        n = 3 if loto == "numbers3" else 4
        digits = df["抽選数字"].astype(str).str.extract(r"([0-9]+)", expand=False).fillna("")
        digits = digits.str[-n:].str.zfill(n)
        for i in range(1, n + 1):
            df[f"N{i}"] = _to_int_series(digits.str.slice(i - 1, i))
        df = df.drop(columns=["抽選数字"])

    rename_nums = {}
    for c in df.columns:
        m = re.fullmatch(r"第(\d+)数字", str(c))
        if m:
            rename_nums[c] = f"N{int(m.group(1))}"
    if rename_nums:
        df = df.rename(columns=rename_nums)

    bonus_map = {}
    for c in df.columns:
        if c == "ボーナス数字":
            bonus_map[c] = "B1"
        else:
            m = re.fullmatch(r"ボーナス数字(\d+)", str(c))
            if m:
                bonus_map[c] = f"B{int(m.group(1))}"
    if bonus_map:
        df = df.rename(columns=bonus_map)

    prize_map = {}
    for c in df.columns:
        m = re.fullmatch(r"(\d+)等口数", str(c))
        if m:
            prize_map[c] = f"PN{int(m.group(1))}"
        m = re.fullmatch(r"(\d+)等賞金", str(c))
        if m:
            prize_map[c] = f"PM{int(m.group(1))}"
    if prize_map:
        df = df.rename(columns=prize_map)

    if "キャリーオーバー" in df.columns:
        df = df.rename(columns={"キャリーオーバー": "CO"})
    elif "キャリーオーバー額" in df.columns:
        df = df.rename(columns={"キャリーオーバー額": "CO"})

    abbrev = {
        "ストレート口数": "STC", "ストレート賞金": "STM",
        "ボックス口数": "BXC", "ボックス賞金": "BXM",
        "セット(ストレート)口数": "SSC", "セット(ストレート)賞金": "SSM",
        "セット(ボックス)口数": "SBC", "セット(ボックス)賞金": "SBM",
        "ミニ口数": "MNC", "ミニ賞金": "MNM",
        "セット（ストレート）口数": "SSC", "セット（ストレート）賞金": "SSM",
        "セット（ボックス）口数": "SBC", "セット（ボックス）賞金": "SBM",
    }
    df = df.rename(columns={k: v for k, v in abbrev.items() if k in df.columns})

    num_pat = re.compile(r"^(N\d+|PN\d+|PM\d+|B\d+|CO|STC|STM|BXC|BXM|SSC|SSM|SBC|SBM|MNC|MNM)$")
    for c in [c for c in df.columns if num_pat.match(str(c))]:
        df[c] = _to_int_series(df[c])

    if "開催回" in df.columns:
        df = df.drop(columns=["開催回"])

    return df


def wide_to_long(df: pd.DataFrame, loto: str) -> pd.DataFrame:
    df = df.copy()
    n_cols = sorted([c for c in df.columns if re.fullmatch(r"N\d+", str(c))], key=lambda x: int(str(x)[1:]))
    if not n_cols:
        raise RuntimeError(f"[{loto}] N列が見つかりません: cols={list(df.columns)}")

    id_vars = [c for c in df.columns if c not in n_cols]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=n_cols,
        var_name="unique_id",
        value_name="y",
    )
    long_df["loto"] = loto
    long_df["unique_id"] = long_df["unique_id"].astype(str).str.upper()
    long_df["y"] = _to_int_series(long_df["y"]).fillna(0).astype("Int64")
    
    base = ["loto", "ds", "unique_id", "y"]
    rest = [c for c in long_df.columns if c not in base]
    return long_df[base + rest]


def add_hist_prefix_and_split(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_long.copy()
    rename_map = {}
    for c in df.columns:
        if c in BASE_COLS:
            continue
        cname = str(c)
        if cname.startswith("hist_"):
            rename_map[c] = cname.lower()
        else:
            rename_map[c] = f"hist_{cname}".lower()
    df = df.rename(columns=rename_map)

    hist_cols = [c for c in df.columns if str(c).startswith("hist_")]
    df_base = df[BASE_COLS].copy()
    df_hist = df[KEY_COLS + hist_cols].copy()

    if hist_cols:
        for c in hist_cols:
            if pd.api.types.is_numeric_dtype(df_hist[c]):
                df_hist[c] = df_hist[c].fillna(0)
            else:
                df_hist[c] = df_hist[c].fillna("")
    return df_base, df_hist


# --------------------------
# PostgreSQL (psql) IO
# --------------------------
def get_env_conn(args: argparse.Namespace) -> Dict[str, str]:
    """引数または環境変数から接続情報を構築"""
    host = args.host or os.getenv("PGHOST") or os.getenv("PG_HOST") or "127.0.0.1"
    port = args.port or os.getenv("PGPORT") or os.getenv("PG_PORT") or "5432"
    db = args.db or os.getenv("PGDATABASE") or os.getenv("PG_DB") or "loto_db"
    user = args.user or os.getenv("PGUSER") or os.getenv("PG_USER") or "loto_user"
    password = args.password or os.getenv("PGPASSWORD") or os.getenv("PG_PASSWORD")
    schema = args.schema or os.getenv("PGSCHEMA") or os.getenv("PG_SCHEMA") or "public"

    env = {"host": host, "port": port, "db": db, "user": user, "schema": schema}
    if password:
        env["password"] = password
    return env


def run_psql(sql: str, env: Dict[str, str], psql_bin: str = "psql", capture: bool = False) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e["PGHOST"] = env["host"]
    e["PGPORT"] = str(env["port"])
    e["PGDATABASE"] = env["db"]
    e["PGUSER"] = env["user"]
    e["PG_SCHEMA"] = env["schema"]
    if env.get("password"):
        e["PGPASSWORD"] = env["password"]
    else:
        e.pop("PGPASSWORD", None)

    e["PAGER"] = "cat"
    e["PSQLRC"] = "/dev/null"

    args = [psql_bin, "-v", "ON_ERROR_STOP=1"]
    args += ["-v", f"schema={env['schema']}"]

    if capture:
        args += ["-A", "-t", "-F", ","]
    return subprocess.run(
        args,
        input=sql,
        text=True,
        env=e,
        capture_output=capture,
        check=True,
    )


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ensure_schema(env: Dict[str, str], psql_bin: str) -> None:
    run_psql(f"CREATE SCHEMA IF NOT EXISTS {qident(env['schema'])};\n", env, psql_bin=psql_bin)


def drop_tables(schema: str, base_table: str, hist_table: str, env: Dict[str, str], psql_bin: str) -> None:
    sql = f"""
DROP TABLE IF EXISTS {qident(schema)}.{qident(base_table)};
DROP TABLE IF EXISTS {qident(schema)}.{qident(hist_table)};
"""
    run_psql(sql, env, psql_bin=psql_bin)


def create_tables(schema: str, base_table: str, hist_table: str, hist_cols: List[str], env: Dict[str, str], psql_bin: str) -> None:
    base_sql = f"""
CREATE TABLE IF NOT EXISTS {qident(schema)}.{qident(base_table)} (
  loto        text        NOT NULL,
  ds          date        NOT NULL,
  unique_id   text        NOT NULL,
  y           bigint      NOT NULL,
  exec_ts     timestamp   NOT NULL,
  updated_ts  timestamp   NOT NULL,
  proc_seconds double precision NOT NULL
);
"""
    hist_defs = []
    for c in hist_cols:
        hist_defs.append(f"{qident(c)} bigint NOT NULL DEFAULT 0")
    hist_extra = (",\n  " + ",\n  ".join(hist_defs)) if hist_defs else ""

    hist_sql = f"""
CREATE TABLE IF NOT EXISTS {qident(schema)}.{qident(hist_table)} (
  loto        text        NOT NULL,
  ds          date        NOT NULL,
  unique_id   text        NOT NULL,
  exec_ts     timestamp   NOT NULL,
  updated_ts  timestamp   NOT NULL,
  proc_seconds double precision NOT NULL
  {hist_extra}
);
"""
    # 【重要】 ON CONFLICT を動作させるために UNIQUE INDEX を作成する
    idx_sql = f"""
CREATE INDEX IF NOT EXISTS {qident(f"idx_{base_table}_loto_ds")} ON {qident(schema)}.{qident(base_table)} (loto, ds);
CREATE INDEX IF NOT EXISTS {qident(f"idx_{base_table}_exec_ts")} ON {qident(schema)}.{qident(base_table)} (exec_ts);
CREATE INDEX IF NOT EXISTS {qident(f"idx_{hist_table}_loto_ds")} ON {qident(schema)}.{qident(hist_table)} (loto, ds);
CREATE INDEX IF NOT EXISTS {qident(f"idx_{hist_table}_exec_ts")} ON {qident(schema)}.{qident(hist_table)} (exec_ts);

CREATE UNIQUE INDEX IF NOT EXISTS {qident(f"uidx_{base_table}_keys")} ON {qident(schema)}.{qident(base_table)} (loto, ds, unique_id, y);
CREATE UNIQUE INDEX IF NOT EXISTS {qident(f"uidx_{hist_table}_keys")} ON {qident(schema)}.{qident(hist_table)} (loto, ds, unique_id);
"""
    run_psql(base_sql + hist_sql + idx_sql, env, psql_bin=psql_bin)


def fetch_existing_columns(schema: str, table: str, env: Dict[str, str], psql_bin: str) -> List[str]:
    sql = (
        "SELECT column_name "
        "FROM information_schema.columns "
        f"WHERE table_schema='{schema}' AND table_name='{table}' "
        "ORDER BY ordinal_position;"
    )
    cp = run_psql(sql, env, psql_bin=psql_bin, capture=True)
    out = [line.strip().split(",")[0] for line in cp.stdout.splitlines() if line.strip()]
    return out


def alter_add_columns(schema: str, table: str, add_cols: List[str], env: Dict[str, str], psql_bin: str) -> None:
    if not add_cols:
        return
    stmts = []
    for c in add_cols:
        stmts.append(
            f"ALTER TABLE {qident(schema)}.{qident(table)} ADD COLUMN IF NOT EXISTS {qident(c)} bigint NOT NULL DEFAULT 0;"
        )
    run_psql("\n".join(stmts) + "\n", env, psql_bin=psql_bin)


def copy_from_csv(schema, table, cols, csv_path, env, key_cols, psql_bin="psql"):
    """
    key_cols: 重複判定（DISTINCT ON / ON CONFLICT）に使用する列のリスト
    """
    stg = f"_{table}_stg"
    nonkey_cols = [c for c in cols if c not in key_cols]
    if nonkey_cols:
        set_sql = ", ".join([f"{qident(c)} = EXCLUDED.{qident(c)}" for c in nonkey_cols])
        on_conflict = f"DO UPDATE SET {set_sql}"
    else:
        on_conflict = "DO NOTHING"

    keys_sql = ", ".join([qident(c) for c in key_cols])
    cols_sql = ", ".join([qident(c) for c in cols])
    tgt = f"{qident(schema)}.{qident(table)}"
    
    order_tail = ""
    if "exec_ts" in cols:
        order_tail = ", " + qident("exec_ts") + " DESC NULLS LAST"

    cmd = f"""
    DROP TABLE IF EXISTS {qident(stg)};
    CREATE TEMP TABLE {qident(stg)} (LIKE {tgt});
    \\copy {qident(stg)} ({cols_sql}) FROM '{csv_path}' WITH (FORMAT csv);
    INSERT INTO {tgt} ({cols_sql})
    SELECT {cols_sql}
    FROM (
      SELECT DISTINCT ON ({keys_sql}) {cols_sql}
      FROM {qident(stg)}
      ORDER BY {keys_sql}{order_tail}
    ) s
    ON CONFLICT ({keys_sql}) {on_conflict};
    """
    run_psql(cmd + "\n", env, psql_bin=psql_bin)


def run_deduplication(env: Dict[str, str], schema: str, table: str, psql_bin: str) -> None:
    print(f"--- Running Deduplication on {schema}.{table} ---")
    sql = f"""
    DELETE FROM {qident(schema)}.{qident(table)} a
    USING {qident(schema)}.{qident(table)} b
    WHERE a.ctid < b.ctid
      AND a.loto = b.loto
      AND a.ds = b.ds
      AND a.unique_id = b.unique_id
      AND a.y = b.y;
    """
    run_psql(sql, env, psql_bin=psql_bin)
    print("--- Deduplication Done ---")


def main() -> int:
    ap = argparse.ArgumentParser(description="Loto Life CSV to PostgreSQL ETL")
    
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--mode", choices=["append", "overwrite"], default="append")
    ap.add_argument("--base-table", default="loto_base")
    ap.add_argument("--hist-table", default="loto_hist_2")
    ap.add_argument("--lotos", default="mini,loto6,loto7,bingo5,numbers3,numbers4")
    ap.add_argument("--psql-bin", default="psql")
    ap.add_argument("--tmp-dir", default=None)

    args = ap.parse_args()
    env = get_env_conn(args)
    
    schema = env["schema"]
    base_table = args.base_table
    hist_table = args.hist_table

    lotos = [x.strip() for x in args.lotos.split(",") if x.strip()]
    for loto in lotos:
        if loto not in URLS:
            raise RuntimeError(f"未知の loto: {loto}")

    t0 = time.time()
    exec_ts = datetime.now()
    updated_ts = exec_ts

    print(f"Start ETL mode={args.mode} target={schema}.{base_table}/{hist_table}")

    long_list: List[pd.DataFrame] = []
    for loto in lotos:
        print(f"Processing {loto}...")
        df_raw = read_loto_csv(URLS[loto])
        df_norm = normalize_loto_df(df_raw, loto=loto)
        df_long = wide_to_long(df_norm, loto=loto)
        long_list.append(df_long)

    df_all = pd.concat(long_list, ignore_index=True)
    df_base, df_hist = add_hist_prefix_and_split(df_all)

    proc_seconds = float(time.time() - t0)
    df_base["exec_ts"] = exec_ts
    df_base["updated_ts"] = updated_ts
    df_base["proc_seconds"] = proc_seconds
    df_hist["exec_ts"] = exec_ts
    df_hist["updated_ts"] = updated_ts
    df_hist["proc_seconds"] = proc_seconds

    df_base.columns = [c.lower() for c in df_base.columns]
    df_hist.columns = [c.lower() for c in df_hist.columns]

    hist_cols = sorted([c for c in df_hist.columns if c.startswith("hist_")])
    for c in hist_cols:
        if pd.api.types.is_numeric_dtype(df_hist[c]):
            df_hist[c] = df_hist[c].fillna(0).astype("Int64")
        else:
            df_hist[c] = df_hist[c].fillna("")
    df_base["y"] = _to_int_series(df_base["y"]).fillna(0).astype("Int64")

    ensure_schema(env, args.psql_bin)

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="loto_pg_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_csv = tmp_dir / "loto_base.csv"
    hist_csv = tmp_dir / "loto_hist.csv"

    if args.mode == "overwrite":
        drop_tables(schema, base_table, hist_table, env, args.psql_bin)
        create_tables(schema, base_table, hist_table, hist_cols, env, args.psql_bin)
    else:
        create_tables(schema, base_table, hist_table, hist_cols, env, args.psql_bin)
        existing = fetch_existing_columns(schema, hist_table, env, args.psql_bin)
        need_add = [c for c in hist_cols if c not in set(existing)]
        alter_add_columns(schema, hist_table, need_add, env, args.psql_bin)

    existing_hist_cols = fetch_existing_columns(schema, hist_table, env, args.psql_bin)
    existing_hist_only = [c for c in existing_hist_cols if c.startswith("hist_")]
    for c in existing_hist_only:
        if c not in df_hist.columns:
            df_hist[c] = 0
            
    base_cols = ["loto", "ds", "unique_id", "y", "exec_ts", "updated_ts", "proc_seconds"]
    hist_cols_order = ["loto", "ds", "unique_id"] + existing_hist_only + ["exec_ts", "updated_ts", "proc_seconds"]

    df_base_out = df_base[base_cols].copy()
    df_hist_out = df_hist[hist_cols_order].copy()

    df_base_out.to_csv(base_csv, index=False, header=False, na_rep="\\N")
    df_hist_out.to_csv(hist_csv, index=False, header=False, na_rep="\\N")

    # 9) COPY投入 (呼び出し時に key_cols を指定する形に修正済み)
    # baseは y も含めた4要素を一意とする
    copy_from_csv(schema, base_table, base_cols, base_csv, env, 
                  key_cols=["loto", "ds", "unique_id", "y"], 
                  psql_bin=args.psql_bin)
    
    # histは y を含まない3要素を一意とする
    copy_from_csv(schema, hist_table, hist_cols_order, hist_csv, env, 
                  key_cols=["loto", "ds", "unique_id"], 
                  psql_bin=args.psql_bin)

    print("[OK] saved to PostgreSQL")
    print(f"  base: {schema}.{base_table} rows={len(df_base_out)}")
    print(f"  hist: {schema}.{hist_table} rows={len(df_hist_out)}")

    run_deduplication(env, schema, base_table, args.psql_bin)

    print(f"  exec_ts={exec_ts.isoformat(sep=' ')} proc_seconds={proc_seconds:.3f}")
    
    if not args.tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
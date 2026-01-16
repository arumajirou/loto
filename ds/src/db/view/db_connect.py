# /mnt/e/env/ts/tslib/ds/src/db/view/db_connect.py

from __future__ import annotations

import html
import importlib.util
from typing import List, Dict, Tuple, Any

import pandas as pd
from sqlalchemy import create_engine, text, inspect


def _sorted_alpha(items: List[str]) -> List[str]:
    return sorted([str(x) for x in items], key=lambda s: s.casefold())


def _quote_ident(ident: str) -> str:
    # SQL識別子を安全にダブルクォート（" を "" に）
    return '"' + str(ident).replace('"', '""') + '"'


def _detect_pg_dialect() -> str:
    # psycopg(=psycopg3) / psycopg2 を自動判定
    if importlib.util.find_spec("psycopg") is not None:
        return "psycopg"
    if importlib.util.find_spec("psycopg2") is not None:
        return "psycopg2"
    # どちらも無いなら明示的に落とす（迷子防止）
    raise ModuleNotFoundError(
        "PostgreSQLドライバが見つかりません。次のどちらかを入れてください:\n"
        "  - pip install \"psycopg[binary]\"   (psycopg3)\n"
        "  - pip install psycopg2-binary      (psycopg2)\n"
        "  - conda install -c conda-forge psycopg\n"
    )


# ==========================================
# 1. DB接続・データ取得ロジック
# ==========================================

def get_engine(host, port, db, user, password):
    """DBエンジンを作成して返す（psycopg/psycopg2 自動判定）"""
    dialect = _detect_pg_dialect()
    db_url = f"postgresql+{dialect}://{user}:{password}@{host}:{port}/{db}"
    try:
        engine = create_engine(db_url, future=True, pool_pre_ping=True)
        return engine
    except Exception as e:
        raise ConnectionError(f"PostgreSQL接続エラー: {e}")


def get_db_info(engine, db_name: str) -> Tuple[str, str, str]:
    """DB全体のサイズ、バージョン、接続ユーザー情報を取得"""
    try:
        with engine.connect() as conn:
            query_size = text("SELECT pg_size_pretty(pg_database_size(:db));")
            db_size = conn.execute(query_size, {"db": db_name}).scalar()

            query_version = text("SELECT version();")
            db_version = conn.execute(query_version).scalar()  # フル文字列

            query_user = text("SELECT current_user;")
            db_user = conn.execute(query_user).scalar()

            return str(db_size), str(db_version), str(db_user)
    except Exception as e:
        return "N/A", f"Error: {e}", "N/A"


def get_schema_list(engine) -> list:
    """スキーマ名リストを取得（大小無視アルファベット順）"""
    try:
        with engine.connect() as conn:
            q = text(
                "SELECT nspname "
                "FROM pg_catalog.pg_namespace "
                "WHERE nspname NOT LIKE 'pg_%' "
                "  AND nspname != 'information_schema';"
            )
            result = conn.execute(q).scalars().all()
            return _sorted_alpha(list(result))
    except Exception as e:
        raise Exception(f"スキーマリスト取得エラー: {e}")


def get_table_list(engine, schema: str) -> List[Dict[str, Any]]:
    """指定スキーマのテーブル名、サイズ、レコード数を取得（名前で大小無視ソート）"""
    try:
        q = text("""
        SELECT
            c.relname AS table_name,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
            pg_size_pretty(pg_relation_size(c.oid)) AS data_size,
            CASE WHEN reltuples = -1 THEN -1 ELSE reltuples::bigint END AS row_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relkind = 'r'
        ORDER BY c.relname;
        """)
        df_tables = pd.read_sql(q, engine, params={"schema": schema})
        recs = df_tables.to_dict("records")
        recs.sort(key=lambda r: str(r.get("table_name", "")).casefold())
        return recs
    except Exception as e:
        raise PermissionError(f"テーブル情報取得エラー: {e}")


def fetch_table_data(engine, schema, table_name, limit: int = 10) -> Tuple[int, pd.DataFrame]:
    """選択されたテーブルのデータと総レコード数を取得"""
    try:
        q_schema = _quote_ident(schema)
        q_table = _quote_ident(table_name)

        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {q_schema}.{q_table}")).scalar()

            df_cols = pd.read_sql(text(f"SELECT * FROM {q_schema}.{q_table} LIMIT 0"), conn)
            order_by_clause = "ORDER BY ds DESC NULLS LAST" if "ds" in df_cols.columns else ""
            df_data = pd.read_sql(
                text(f"SELECT * FROM {q_schema}.{q_table} {order_by_clause} LIMIT :limit"),
                conn,
                params={"limit": int(limit)},
            )

            return int(count), df_data

    except Exception as e:
        return -1, pd.DataFrame({"Error": [str(e)]})


def _one_line(s: str) -> str:
    return " ".join(str(s).split())


def _fmt_value(v: Any, max_chars: int = 140) -> str:
    if v is None:
        s = "NULL"
    else:
        s = _one_line(v)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s


def fetch_unique_values(engine, schema: str, table_name: str, limit: int = 50) -> Dict[str, str]:
    """
    カラムごとにユニーク値（頻度順 上位limit）を “HTMLの箇条書き” で返す
    """
    try:
        unique_info: Dict[str, str] = {}
        q_schema = _quote_ident(schema)
        q_table = _quote_ident(table_name)

        with engine.connect() as conn:
            # カラム名を取得（順序維持）
            df_cols = pd.read_sql(text(f"SELECT * FROM {q_schema}.{q_table} LIMIT 0"), conn)

            for col in df_cols.columns:
                q_col = _quote_ident(col)

                total_unique_count = conn.execute(
                    text(f"SELECT COUNT(DISTINCT {q_col}) FROM {q_schema}.{q_table}")
                ).scalar() or 0

                rows = conn.execute(
                    text(f"""
                        SELECT {q_col} AS val, COUNT(*) AS n
                        FROM {q_schema}.{q_table}
                        GROUP BY {q_col}
                        ORDER BY n DESC
                        LIMIT :limit
                    """),
                    {"limit": int(limit)},
                ).fetchall()

                li_items = []
                for val, n in rows:
                    show = _fmt_value(val, max_chars=140)
                    li_items.append(
                        f"<li style='margin:2px 0;'>"
                        f"<span>{html.escape(show)}</span> "
                        f"<span style='color:#888'>(freq: {int(n):,})</span>"
                        f"</li>"
                    )
                if not li_items:
                    li_items.append("<li>(なし)</li>")

                html_output = f"""
                <details style="margin-top: 10px;">
                    <summary>
                        <strong style='color:#007bff;'>{html.escape(str(col))}</strong>
                        (ユニーク数: {int(total_unique_count):,})
                    </summary>
                    <div style="padding: 10px 15px; background-color: #f9f9f9; border: 1px solid #eee; border-radius: 6px; margin-top:6px;">
                        <ul style="margin:0; padding-left: 18px;">
                            {''.join(li_items)}
                        </ul>
                    </div>
                </details>
                """
                unique_info[str(col)] = html_output

        return unique_info

    except Exception as e:
        return {"Error": f"ユニーク値の取得エラー: {str(e)}"}


def drop_table(engine, schema: str, table_name: str):
    """テーブルを削除する"""
    try:
        q_schema = _quote_ident(schema)
        q_table = _quote_ident(table_name)
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {q_schema}.{q_table} CASCADE;"))
        return True
    except Exception as e:
        raise Exception(f"テーブル削除エラー: {str(e)}")


def get_table_schema(engine, schema: str, table_name: str) -> pd.DataFrame:
    """SQLAlchemy Inspectorでテーブルスキーマ情報を取得しDataFrame化"""
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)

        schema_data = []
        for col in columns:
            col_type_str = str(col["type"])
            schema_data.append(
                {
                    "カラム名": col["name"],
                    "データ型": col_type_str,
                    "最大長": col.get("length", "N/A")
                    if "TEXT" not in col_type_str.upper() and "VARCHAR" not in col_type_str.upper()
                    else "N/A",
                    "NULL許容": "YES" if col.get("nullable", True) else "NO",
                    "デフォルト値": col.get("default"),
                }
            )

        df_schema = pd.DataFrame(schema_data)
        return df_schema.fillna("None")

    except Exception as e:
        raise Exception(f"テーブルスキーマ取得エラー: {e}")

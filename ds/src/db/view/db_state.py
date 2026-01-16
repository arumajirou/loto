# /mnt/e/env/ts/tslib/ds/src/db/view/db_state.py

from __future__ import annotations

from sqlalchemy import text
from typing import List, Dict, Any, Optional

# ✅ 同じパッケージ（db.view）から相対import（握りつぶさない）
from .db_connect import (
    get_engine,
    get_db_info,
    get_schema_list,
    get_table_list,
    fetch_table_data,
    fetch_unique_values,
    drop_table,
    get_table_schema,
)


def _sorted_alpha(items: List[str]) -> List[str]:
    return sorted([str(x) for x in items], key=lambda s: s.casefold())


class DBState:
    """データベースの接続状態とデータ操作を管理するクラス"""

    def __init__(self, host, port, user, password, initial_db):
        self.host = host
        self.port = str(port)
        self.user = user
        self.password = password
        self.current_db = initial_db
        self.engine = None

        self.db_size = "N/A"
        self.db_version = "N/A"
        self.db_conn_user = "N/A"

    def connect(self, dbname: Optional[str] = None):
        if dbname:
            self.current_db = dbname

        if self.engine:
            self.engine.dispose()

        # ✅ ドライバ自動判定込みの get_engine を使う
        self.engine = get_engine(self.host, self.port, self.current_db, self.user, self.password)

        # ✅ get_db_info は必ず import 済み
        self.db_size, self.db_version, self.db_conn_user = get_db_info(self.engine, self.current_db)
        return True

    def fetch_all_databases(self) -> List[str]:
        """
        DB一覧を postgres に接続して取得（大小無視アルファベット順）
        """
        temp_engine = get_engine(self.host, self.port, "postgres", self.user, self.password)
        try:
            with temp_engine.connect() as conn:
                q = text("SELECT datname FROM pg_database WHERE datistemplate = false;")
                dbs = [r[0] for r in conn.execute(q).fetchall()]
            if self.current_db not in dbs:
                dbs.append(self.current_db)
            return _sorted_alpha(dbs)
        except Exception:
            return [self.current_db]
        finally:
            temp_engine.dispose()

    def get_schemas(self) -> List[str]:
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return get_schema_list(self.engine)

    def get_tables(self, schema) -> List[Dict[str, Any]]:
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return get_table_list(self.engine, schema)

    def get_table_data(self, schema, table, limit: int = 10):
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return fetch_table_data(self.engine, schema, table, limit=limit)

    def get_column_schema(self, schema, table):
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return get_table_schema(self.engine, schema, table)

    # ✅ ここが今回のバグ修正: limitを受け取れるようにする
    def get_unique_values(self, schema, table, limit: int = 50, **_kwargs):
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return fetch_unique_values(self.engine, schema, table, limit=limit)

    def delete_table(self, schema, table):
        if not self.engine:
            raise RuntimeError("DB未接続です")
        return drop_table(self.engine, schema, table)

# /mnt/e/env/ts/tslib/ds/src/db/view/db_app.py

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from IPython.display import display, clear_output, HTML

from .db_state import DBState
from .db_ui import DBViewerUI


def _sorted_alpha(items: List[str]) -> List[str]:
    return sorted([str(x) for x in items], key=lambda s: s.casefold())


def _fmt_rows(x: Any) -> str:
    try:
        # reltuples は float のことがある
        xi = int(float(x))
        if xi < 0:
            return "?"
        return f"{xi:,}"
    except Exception:
        return "?"


class DBViewerApp:
    """
    - Tables 選択リスト: テーブル名のみ
    - 追加で “リッチ表” (rows/size) を表示できるモード
    - Unique Values は details/summary + 箇条書きHTML
    """

    def __init__(
        self,
        host: str,
        port: str,
        dbname: str,
        user: str,
        password: str,
        default_schema: str = "public",
    ) -> None:
        self.state = DBState(host, port, user, password, initial_db=dbname)
        self.state.connect(dbname)

        self.default_schema = default_schema
        self.ui = DBViewerUI()

        self._suspend = False
        self._tables_info_cache: List[Dict[str, Any]] = []

        self._wire_events()
        self._bootstrap()

    # ----------------------------
    # wiring / bootstrap
    # ----------------------------
    def _wire_events(self) -> None:
        # DB / schema
        self.ui.db_select.observe(self._on_db_change, names="value")
        self.ui.schema_select.observe(self._on_schema_change, names="value")

        # 表示モード
        self.ui.tables_view_mode.observe(self._on_tables_view_mode_change, names="value")

        # ✅ 常時表示バー更新トリガ（テーブル選択/タブ切替）
        self.ui.data_table_select.observe(self._on_context_change, names="value")
        self.ui.schema_table_select.observe(self._on_context_change, names="value")
        self.ui.unique_table_select.observe(self._on_context_change, names="value")
        self.ui.delete_table_dropdown.observe(self._on_context_change, names="value")
        self.ui.tab_widget.observe(self._on_context_change, names="selected_index")

        # actions
        self.ui.data_load_button.on_click(self._on_preview_data)
        self.ui.schema_load_button.on_click(self._on_show_schema)
        self.ui.unique_analyze_button.on_click(self._on_analyze_unique)
        self.ui.delete_confirm_button.on_click(self._on_delete_table)

    def _bootstrap(self) -> None:
        self._suspend = True
        try:
            dbs = self.state.fetch_all_databases()
            self.ui.db_select.options = dbs
            self.ui.db_select.value = (
                self.state.current_db if self.state.current_db in dbs else (dbs[0] if dbs else None)
            )

            self.ui.update_header(
                self.state.current_db,
                self.state.db_size,
                self.state.db_version,
                self.state.db_conn_user,
            )

            schemas = self.state.get_schemas()
            self.ui.schema_select.options = schemas

            sch = self.default_schema if self.default_schema in schemas else (schemas[0] if schemas else None)
            self.ui.schema_select.value = sch

            if sch:
                self._refresh_tables(sch)

            # ✅ 初期状態でも表示（常時バー）
            self._update_context_bar()
        finally:
            self._suspend = False

    # ----------------------------
    # table refresh / rich table render
    # ----------------------------
    def _refresh_tables(self, schema: str) -> None:
        tables_info = self.state.get_tables(schema)  # list[dict]
        self._tables_info_cache = tables_info

        names = _sorted_alpha([t.get("table_name", "") for t in tables_info if t.get("table_name")])

        # 選択肢はテーブル名だけ
        self.ui.data_table_select.options = names
        self.ui.schema_table_select.options = names
        self.ui.unique_table_select.options = names
        self.ui.delete_table_dropdown.options = names

        # 初期選択
        first = names[0] if names else None
        if first:
            self.ui.schema_table_select.value = first
            self.ui.unique_table_select.value = first
            self.ui.delete_table_dropdown.value = first
            self.ui.data_table_select.value = (first,)  # SelectMultiple は tuple

        # 表示モードに応じてメタ表を描画
        self._render_tables_meta()

        # ✅ tables再生成後も更新
        self._update_context_bar()

    def _render_tables_meta(self) -> None:
        """Tablesのメタ情報（rows/size）を“リッチ表”で表示する"""
        mode = self.ui.tables_view_mode.value

        with self.ui.tables_meta_output:
            clear_output()

            if mode != "rich":
                return

            if not self._tables_info_cache:
                display(HTML("<i>テーブル情報がありません</i>"))
                return

            df = pd.DataFrame(self._tables_info_cache)

            keep = [c for c in ["table_name", "row_count", "total_size", "data_size"] if c in df.columns]
            df = df[keep].copy()

            if "table_name" in df.columns:
                df["table_name"] = df["table_name"].astype(str)

            if "row_count" in df.columns:
                df["row_count"] = df["row_count"].apply(_fmt_rows)

            rename = {
                "table_name": "Table",
                "row_count": "Rows(approx)",
                "total_size": "Total",
                "data_size": "Data",
            }
            df = df.rename(columns=rename)

            if "Table" in df.columns:
                df = df.sort_values("Table", key=lambda s: s.astype(str).str.casefold())

            display(HTML("<h4 style='margin:10px 0 5px 0;'>📋 Tables (rich)</h4>"))
            display(df)

    # ----------------------------
    # context bar (dynamic)
    # ----------------------------
    def _on_context_change(self, _change) -> None:
        if self._suspend:
            return
        self._update_context_bar()

    def _update_context_bar(self) -> None:
        """選択中のschema/table と 接続情報を、常にヘッダー直下へ表示"""
        schema = self.ui.schema_select.value or "-"

        idx = self.ui.tab_widget.selected_index
        idx = 0 if idx is None else int(idx)
        tab_name_map = {
            0: "📊 Data Preview",
            1: "📋 Schema & Size",
            2: "✨ Unique Values",
            3: "🗑️ Delete",
        }
        tab_name = tab_name_map.get(idx, f"Tab({idx})")

        # 「いま見ているタブ」での“選択テーブル”を採用
        table = "-"
        if idx == 0:
            tables = list(self.ui.data_table_select.value or [])
            if tables:
                # 長くなりすぎない表示（UIで視認性優先）
                if len(tables) <= 3:
                    table = ", ".join(tables)
                else:
                    table = ", ".join(tables[:3]) + f" (+{len(tables) - 3})"
        elif idx == 1:
            table = self.ui.schema_table_select.value or "-"
        elif idx == 2:
            table = self.ui.unique_table_select.value or "-"
        elif idx == 3:
            table = self.ui.delete_table_dropdown.value or "-"

        # passwordを出さない接続文字列
        conn = f"postgresql://{self.state.user}@{self.state.host}:{self.state.port}/{self.state.current_db}"

        # driver名（例: postgresql+psycopg）
        driver = "-"
        try:
            if getattr(self.state, "engine", None) is not None:
                driver = str(getattr(self.state.engine.url, "drivername", "")) or "-"
        except Exception:
            driver = "-"

        self.ui.update_context(
            conn=conn,
            driver=driver,
            schema=str(schema),
            tab=str(tab_name),
            table=str(table),
            login_user=str(self.state.user),
            db_user=str(self.state.db_conn_user),
        )

    # ----------------------------
    # event handlers
    # ----------------------------
    def _on_tables_view_mode_change(self, _change) -> None:
        if self._suspend:
            return
        self._render_tables_meta()
        self._update_context_bar()

    def _on_db_change(self, change) -> None:
        if self._suspend:
            return
        new_db = change["new"]
        if not new_db:
            return

        self._suspend = True
        try:
            self.ui.clear_all_outputs()
            self.state.connect(new_db)
            self.ui.update_header(self.state.current_db, self.state.db_size, self.state.db_version, self.state.db_conn_user)

            schemas = self.state.get_schemas()
            self.ui.schema_select.options = schemas
            sch = self.default_schema if self.default_schema in schemas else (schemas[0] if schemas else None)
            self.ui.schema_select.value = sch
            if sch:
                self._refresh_tables(sch)

            self._update_context_bar()
        finally:
            self._suspend = False

    def _on_schema_change(self, change) -> None:
        if self._suspend:
            return
        sch = change["new"]
        if not sch:
            return

        self._suspend = True
        try:
            self.ui.clear_all_outputs()
            self._refresh_tables(sch)
            self._update_context_bar()
        finally:
            self._suspend = False

    def _on_preview_data(self, _btn) -> None:
        sch = self.ui.schema_select.value
        tables = list(self.ui.data_table_select.value or [])
        if not sch or not tables:
            with self.ui.data_output:
                clear_output()
                display(HTML("<b style='color:#b00'>schema/table を選択してください</b>"))
            return

        with self.ui.data_output:
            clear_output()
            for t in tables:
                count, df = self.state.get_table_data(sch, t, limit=10)
                display(HTML(f"<h4 style='margin:10px 0 5px 0;'>📊 {sch}.{t} (rows: {count:,})</h4>"))
                display(df)

    def _on_show_schema(self, _btn) -> None:
        sch = self.ui.schema_select.value
        t = self.ui.schema_table_select.value
        if not sch or not t:
            with self.ui.schema_output:
                clear_output()
                display(HTML("<b style='color:#b00'>schema/table を選択してください</b>"))
            return

        with self.ui.schema_output:
            clear_output()
            df_schema = self.state.get_column_schema(sch, t)
            display(HTML(f"<h4 style='margin:10px 0 5px 0;'>📋 Schema: {sch}.{t}</h4>"))
            display(df_schema)

    def _on_analyze_unique(self, _btn) -> None:
        sch = self.ui.schema_select.value
        t = self.ui.unique_table_select.value
        if not sch or not t:
            with self.ui.unique_output:
                clear_output()
                display(HTML("<b style='color:#b00'>schema/table を選択してください</b>"))
            return

        self.ui.unique_analyze_button.disabled = True
        self.ui.unique_analyze_button.description = "Analyzing..."

        try:
            with self.ui.unique_output:
                clear_output()
                display(HTML(f"<h4 style='margin:10px 0 5px 0;'>✨ Unique Values: {sch}.{t}</h4>"))
                unique_info = self.state.get_unique_values(sch, t, limit=50)

                if "Error" in unique_info:
                    display(HTML(f"<b style='color:#b00'>{unique_info['Error']}</b>"))
                    return

                html_all = "".join(unique_info.values())
                display(HTML(f"<div style='max-height: 520px; overflow:auto; padding-right:6px;'>{html_all}</div>"))
        finally:
            self.ui.unique_analyze_button.disabled = False
            self.ui.unique_analyze_button.description = "Analyze Unique"

    def _on_delete_table(self, _btn) -> None:
        sch = self.ui.schema_select.value
        t = self.ui.delete_table_dropdown.value
        if not sch or not t:
            with self.ui.delete_output:
                clear_output()
                display(HTML("<b style='color:#b00'>schema/table を選択してください</b>"))
            return

        with self.ui.delete_output:
            clear_output()
            try:
                self.state.delete_table(sch, t)
                display(HTML(f"<b style='color:green'>Deleted:</b> {sch}.{t}"))
                self._suspend = True
                try:
                    self._refresh_tables(sch)
                finally:
                    self._suspend = False
            except Exception as e:
                display(HTML(f"<b style='color:#b00'>削除エラー:</b> {e}"))
            finally:
                # ✅ 削除後も常時バーを最新化
                self._update_context_bar()

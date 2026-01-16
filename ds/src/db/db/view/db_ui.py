# /mnt/e/env/ts/tslib/ds/src/db/view/db_ui.py

import html
import ipywidgets as widgets
from IPython.display import display, clear_output


class DBViewerUI:
    """UIコンポーネントとレイアウト"""

    def __init__(self):
        # Top
        self.db_select = widgets.Dropdown(description="Database:", layout=widgets.Layout(width="320px"))
        self.schema_select = widgets.Dropdown(description="Schema:", layout=widgets.Layout(width="240px"))

        # Header
        self.header_output = widgets.Output()

        # ✅ 常時表示バー（接続/選択情報）
        self.context_html = widgets.HTML()

        # ---- Data Tab ----
        # 表示モード: テーブル名のみ / リッチ表
        self.tables_view_mode = widgets.ToggleButtons(
            options=[("テーブル名のみ", "name"), ("リッチ表", "rich")],
            value="name",
            description="表示:",
            layout=widgets.Layout(width="300px"),
        )

        self.data_table_select = widgets.SelectMultiple(
            description="Tables:",
            layout=widgets.Layout(width="100%", height="180px"),
        )
        self.data_load_button = widgets.Button(description="Preview Data", button_style="primary", icon="table")

        # ✅ テーブル一覧（メタ情報）を出す場所
        self.tables_meta_output = widgets.Output(layout=widgets.Layout(width="100%"))

        self.data_output = widgets.Output()

        # ---- Schema Tab ----
        self.schema_table_select = widgets.Dropdown(description="Table:", layout=widgets.Layout(width="420px"))
        self.schema_load_button = widgets.Button(description="Show Schema", button_style="info", icon="list-alt")
        self.schema_output = widgets.Output()

        # ---- Unique Tab ----
        self.unique_table_select = widgets.Dropdown(description="Table:", layout=widgets.Layout(width="420px"))
        self.unique_analyze_button = widgets.Button(description="Analyze Unique", button_style="warning", icon="bar-chart")
        self.unique_output = widgets.Output()

        # ---- Delete Tab ----
        self.delete_table_dropdown = widgets.Dropdown(description="Target:", layout=widgets.Layout(width="420px"))
        self.delete_confirm_button = widgets.Button(description="Delete Table", button_style="danger", icon="trash")
        self.delete_output = widgets.Output()

        self._build_layout()

    def _build_layout(self):
        top_controls = widgets.HBox(
            [self.db_select, self.schema_select],
            layout=widgets.Layout(margin="0 0 10px 0"),
        )

        data_tab = widgets.VBox(
            [
                widgets.HBox([widgets.Label("Select Tables:"), self.tables_view_mode, self.data_load_button]),
                self.data_table_select,
                self.tables_meta_output,  # ✅ ここに“リッチ表”を表示
                widgets.HTML("<hr>"),
                self.data_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        schema_tab = widgets.VBox(
            [
                widgets.HBox([self.schema_table_select, self.schema_load_button]),
                self.schema_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        unique_tab = widgets.VBox(
            [
                widgets.HBox([self.unique_table_select, self.unique_analyze_button]),
                self.unique_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        delete_tab = widgets.VBox(
            [
                widgets.HTML("<h4 style='color:red;'>⚠️ 危険: テーブル削除エリア</h4>"),
                widgets.HBox([self.delete_table_dropdown, self.delete_confirm_button]),
                self.delete_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        self.tab_widget = widgets.Tab(children=[data_tab, schema_tab, unique_tab, delete_tab])
        self.tab_widget.set_title(0, "📊 Data Preview")
        self.tab_widget.set_title(1, "📋 Schema & Size")
        self.tab_widget.set_title(2, "✨ Unique Values")
        self.tab_widget.set_title(3, "🗑️ Delete")

        self.container = widgets.VBox([self.header_output, self.context_html, top_controls, self.tab_widget])

    def update_header(self, db_name, size, version, user):
        with self.header_output:
            clear_output()
            info_text = f"<b>Size:</b> {size} | <b>Ver:</b> {version} | <b>User:</b> {user}"
            display(
                widgets.HTML(
                    f"""
                <div style='background-color: #f4f4f4; padding: 10px; border-radius: 8px; border: 1px solid #ddd;'>
                    <div style='font-size: 1.2em; font-weight: bold; color: #333; margin-bottom: 5px;'>
                        🗄️ Database Explorer: <span style='color:blue'>{db_name}</span>
                    </div>
                    <div style='font-size: 0.9em; color: #555;'>{info_text}</div>
                </div>
                """
                )
            )


def update_context(
    self,
    *,
    conn: str,
    driver: str,
    schema: str,
    tab: str,
    table: str,
    login_user: str,
    db_user: str,
) -> None:
    """常時表示バーを更新（パスワードは表示しない）"""
    esc = html.escape
    self.context_html.value = f"""
    <div style='background:#ffffff; padding: 8px 10px; border-radius: 8px; border: 1px solid #eee; margin: 8px 0 10px 0;'>
        <div style='font-size: 0.92em; color:#333; line-height:1.45;'>
            <b>Conn:</b> <code>{esc(conn)}</code>
            &nbsp; <span style='color:#999'>|</span> &nbsp;
            <b>Driver:</b> <code>{esc(driver)}</code>
            &nbsp; <span style='color:#999'>|</span> &nbsp;
            <b>Schema:</b> <code>{esc(schema)}</code>
            &nbsp; <span style='color:#999'>|</span> &nbsp;
            <b>Tab:</b> {esc(tab)}
            &nbsp; <span style='color:#999'>|</span> &nbsp;
            <b>Table:</b> <span style='color:#007bff; font-weight:600'>{esc(table)}</span>
            <br>
            <span style='color:#666;'>
                <b>LoginUser:</b> <code>{esc(login_user)}</code>
                &nbsp; <span style='color:#bbb'>/</span> &nbsp;
                <b>DB current_user:</b> <code>{esc(db_user)}</code>
            </span>
        </div>
    </div>
    """

    def clear_all_outputs(self):
        self.data_output.clear_output()
        self.tables_meta_output.clear_output()
        self.schema_output.clear_output()
        self.unique_output.clear_output()
        self.delete_output.clear_output()

    def display(self):
        display(self.container)

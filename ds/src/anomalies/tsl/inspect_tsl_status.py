import os
import pandas as pd
from sqlalchemy import create_engine, text

# --- 設定 ---
DB_USER = "postgres"
DB_PASS = "z"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "time_series_library" # 解析したDB名

# 表示したいクラス名のリスト（エラーが出ている箇所）
TARGET_CLASSES = [
    "Dataset_Custom", 
    "TimesNet", 
    "Autoformer", 
    "Nonstationary_Transformer", 
    "FiLM"
]

def get_db_engine():
    uri = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri)

def show_table_counts(engine):
    """各テーブルの行数を表示してデータが入っているか確認"""
    print("\n📊 --- Database Statistics ---")
    tables = ["modules", "classes", "functions", "arguments"]
    for t in tables:
        try:
            count = pd.read_sql(f"SELECT count(*) FROM {t}", engine).iloc[0, 0]
            print(f"  - {t:<10}: {count:>5} rows")
        except Exception as e:
            print(f"  - {t:<10}: Error ({e})")

def inspect_class_arguments(engine, class_names):
    """指定したクラスの __init__ メソッドの引数を整形して表示"""
    
    # 情報を結合して取得するSQL
    # modules -> classes -> functions(__init__) -> arguments
    query = text("""
        SELECT 
            m.name AS module,
            c.name AS class_name,
            f.name AS method,
            a.position,
            a.name AS arg_name,
            a.default_value,
            a.annotation
        FROM classes c
        JOIN modules m ON c.module_id = m.id
        JOIN functions f ON f.class_id = c.id
        JOIN arguments a ON f.id = a.function_id
        WHERE c.name IN :class_names
          AND f.name = '__init__'
        ORDER BY c.name, a.position
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"class_names": tuple(class_names)})
        
        if df.empty:
            print("\n⚠️  No data found for specified classes.")
            return

        print(f"\n🔍 --- Constructor Arguments for: {', '.join(class_names)} ---")
        
        # クラスごとに見やすく表示
        for cls_name in class_names:
            subset = df[df["class_name"] == cls_name]
            if subset.empty:
                print(f"\n❌ Class '{cls_name}' not found in DB.")
                continue
            
            print(f"\n📘 Class: {cls_name}")
            print(f"   Module: {subset.iloc[0]['module']}")
            print("-" * 80)
            # 必要な列だけ表示
            display_cols = ["position", "arg_name", "default_value", "annotation"]
            print(subset[display_cols].to_string(index=False))
            print("-" * 80)

            # --- エラー原因の特定ヒント ---
            args_list = subset["arg_name"].tolist()
            
            # 1. Dataset_Custom の win_size チェック
            if cls_name == "Dataset_Custom":
                if "win_size" not in args_list:
                    print("   🚨 ALERT: 'win_size' argument is MISSING in __init__!")
                    print("      -> Suggestion: Check if it's renamed to 'seq_len' or removed.")
            
            # 2. Nonstationary_Transformer の p_hidden_dims チェック
            if cls_name == "Nonstationary_Transformer":
                if "p_hidden_dims" not in args_list:
                    print("   🚨 ALERT: 'p_hidden_dims' argument is MISSING in __init__!")

    except Exception as e:
        print(f"Error executing query: {e}")

def main():
    engine = get_db_engine()
    show_table_counts(engine)
    inspect_class_arguments(engine, TARGET_CLASSES)

if __name__ == "__main__":
    main()
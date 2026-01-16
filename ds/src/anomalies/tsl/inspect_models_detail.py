import pandas as pd
from sqlalchemy import create_engine, text

# --- 設定 ---
DB_USER = "postgres"
DB_PASS = "z"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "time_series_library"

# 調査対象のモジュール名（ファイル名）
TARGET_MODULES = [
    "models.TimesNet",
    "models.Autoformer",
    "models.Nonstationary_Transformer",
    "models.FiLM",
    "models.iTransformer",
    "models.Transformer",
    "models.DLinear"
]

def get_db_engine():
    uri = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri)

def inspect_model_classes(engine):
    """
    指定されたモジュール内のクラス（特に 'Model'）とその引数を表示
    """
    print(f"\n🔍 --- Inspecting Classes in Target Modules ---")
    
    # モジュール名からクラスと__init__引数を引くクエリ
    query = text("""
        SELECT 
            m.name AS module_name,
            c.name AS class_name,
            a.position,
            a.name AS arg_name,
            a.default_value,
            a.annotation
        FROM modules m
        JOIN classes c ON c.module_id = m.id
        JOIN functions f ON f.class_id = c.id
        JOIN arguments a ON f.id = a.function_id
        WHERE m.name IN :modules
          AND f.name = '__init__'
          AND c.name = 'Model'  -- TSLの慣例としてクラス名は Model
        ORDER BY m.name, a.position
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"modules": tuple(TARGET_MODULES)})
        
        if df.empty:
            print("⚠️  No 'Model' classes found in specified modules.")
            print("    Check if module names match exactly (e.g. 'models.TimesNet' vs 'models.timesnet').")
            return

        # モジュールごとに表示
        for mod in TARGET_MODULES:
            subset = df[df["module_name"] == mod]
            if subset.empty:
                print(f"\n❌ Module '{mod}': No 'Model' class found or no __init__ args.")
                continue
            
            print(f"\n📦 Module: {mod}")
            print(f"   Class : {subset.iloc[0]['class_name']}")
            print("-" * 90)
            print(subset[["position", "arg_name", "default_value", "annotation"]].to_string(index=False))
            print("-" * 90)
            
            # TimesNetの引数チェック
            if "TimesNet" in mod:
                args = subset["arg_name"].tolist()
                if "num_kernels" in args:
                    print("   ℹ️  'num_kernels' found. Check annotation/default.")
                else:
                    print("   ⚠️  'num_kernels' NOT found in __init__.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    engine = get_db_engine()
    inspect_model_classes(engine)
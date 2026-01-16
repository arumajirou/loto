# /mnt/e/env/ts/tslib/ds/src/anomalies/pypots/naming.py
def get_col_name(pipeline: str, field: str, prefix: str = "hist", library: str = "pypots", sep: str = "__") -> str:
    return f"{prefix}_{library}{sep}{pipeline}{sep}{field}"

def get_output_columns(pipeline: str):
    return {
        "is_anomaly": get_col_name(pipeline, "is_anomaly"),
        "anomaly_score": get_col_name(pipeline, "anomaly_score")
    }
from __future__ import annotations


def sanitize_token(s: str) -> str:
    """Make a token safe for use in SQL column names.

    - Keep alnum and underscore.
    - Replace other chars with underscore.
    """
    out = []
    for ch in str(s):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    # collapse multiple underscores
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def make_pipeline_name(model: str, seq_len: int, stride: int, custom: str | None = None) -> str:
    if custom and str(custom).strip():
        return sanitize_token(custom)
    return sanitize_token(f"{model.lower()}_seq{seq_len}_s{stride}")


def feature_name(
    field: str,
    *,
    hist_prefix: str = "hist",
    lib_prefix: str = "pypots",
    pipeline: str | None = None,
    sep: str = "__",
    include_pipeline: bool = True,
) -> str:
    hist_prefix = sanitize_token(hist_prefix)
    lib_prefix = sanitize_token(lib_prefix)
    field = sanitize_token(field)
    if include_pipeline and pipeline:
        pipeline = sanitize_token(pipeline)
        return f"{hist_prefix}{sep}{lib_prefix}{sep}{pipeline}{sep}{field}"
    return f"{hist_prefix}{sep}{lib_prefix}{sep}{field}"

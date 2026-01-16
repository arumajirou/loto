"""PyPOTS-based anomaly detection runner (DB -> windowing -> fit -> detect -> DB).

This package is intentionally self-contained so it can be used alongside other
libraries (sktime/darts/merlion/...) without column-name collisions.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

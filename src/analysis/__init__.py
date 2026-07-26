"""M2/M3 analysis layer: episode loading, derived fields, impact ranking, prediction.

Episodes are read through an allowlist of columns and metadata keys, so simulator-only
fields never enter this package. The allowlist lives in :mod:`src.analysis.loader`, and
``tests/test_analysis.py`` asserts that no module here so much as names one of them.
"""

from __future__ import annotations

__all__ = ["loader", "impact", "predictor"]

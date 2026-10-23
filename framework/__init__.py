"""Q2 minimal transparent node-level framework (P1).

This package is deliberately dependency-light and CPU-only:

* numpy / pandas / duckdb / scikit-learn
* no torch, no PyG, no DGL
* the one-layer GraphSAGE-style encoder is hand-written in numpy, including
  its analytic backward pass (see :mod:`framework.graph`).

The package is the *framework* artifact for the Q2 methods paper.  It is not a
paper claim and it does not read the frozen test split unless a caller
explicitly sets ``test_policy.allow_test = true``.
"""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA = "q2-framework/1"

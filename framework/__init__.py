"""Public reference implementation of the actor-role attribution contract.

This package is the artefact accompanying the manuscript. It implements one
paper path — an ex ante feature-ownership contract and its audit — not a
reusable framework and not a new estimator. It is deliberately
dependency-light and CPU only:

* numpy / pandas / duckdb / scikit-learn / scipy / matplotlib;
* no torch, no graph neural network library, no GPU path;
* no node-level prediction and no GraphSAGE claim. The unit of analysis is the
  directed dyad-month; actor-level quantities are exact additive projections of
  dyad forecasts, not node forecasts.

The paper path is the dyad-month feature contract, transparent logistic and
HistGradientBoosting models, calibration, exact additive source/target
attribution, actor-level aggregation and cluster-bootstrap inference. The
package does not read the restricted ICEWS panel unless a caller explicitly
provides it; the public release ships aggregate results only.
"""

from __future__ import annotations

__version__ = "0.1.1"
SCHEMA = "q2-framework/1"  # frozen cache/panel format id; not a claim label

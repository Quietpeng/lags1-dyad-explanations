"""Small shared utilities: logging/tee, hashing, host info, JSON, timing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


class Tee:
    """Write stdout/stderr to a run log without losing console output."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.f.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()

    def close(self) -> None:
        self.flush()
        self.f.close()


def json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (datetime,)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=json_default, sort_keys=False)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dir(path: str, pattern: str = "*.py") -> dict[str, str]:
    import glob

    out = {}
    for p in sorted(glob.glob(os.path.join(path, pattern))):
        out[os.path.relpath(p, path)] = sha256_file(p)
    return out


def host_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["ram_gib"] = round(int(line.split()[1]) / 2**20, 1)
                    break
    except OSError:
        pass
    return info


def pkg_versions() -> dict:
    import duckdb
    import pandas as pd
    import sklearn

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
        "scikit_learn": sklearn.__version__,
    }


def seed_all(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def elapsed(t0: float) -> float:
    return round(time.time() - t0, 2)

"""Append-only registry of every configuration tried — including failures.

DSR and PBO are computed over the *full* registry, so the cost of searching is
paid in the statistics rather than hidden.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


class TrialRegistry:
    def __init__(self, root: str | Path, stage: str = "research"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.jsonl"
        self.ret_path = self.root / "returns.csv"
        self.stage = stage
        self._returns: dict[str, pd.Series] = {}
        if self.ret_path.exists():
            df = pd.read_csv(self.ret_path, index_col=0, parse_dates=True)
            self._returns = {c: df[c] for c in df.columns}

    @staticmethod
    def trial_id(config: dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def log(self, config: dict, returns: pd.Series | None = None, metrics: dict | None = None,
            status: str = "ok", error: str | None = None, tags: dict | None = None) -> str:
        tid = self.trial_id(config)
        rec = {"trial_id": tid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": self.stage,
               "status": status, "config": config, "metrics": _jsonable(metrics or {}), "error": error,
               "tags": tags or {}}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if returns is not None:
            self._returns[tid] = returns.rename(tid)
            pd.DataFrame(self._returns).to_csv(self.ret_path)
        return tid

    def run(self, config: dict, fn, tags: dict | None = None):
        """Run ``fn()`` -> (returns, metrics); log success or failure."""
        try:
            returns, metrics = fn()
            tid = self.log(config, returns, metrics, tags=tags)
            return tid, returns, metrics
        except Exception as exc:  # failures are data too
            self.log(config, None, None, status="failed", error=f"{exc!r}\n{traceback.format_exc()[-2000:]}",
                     tags=tags)
            return None, None, None

    def records(self, stage: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        recs = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        return [r for r in recs if stage is None or r["stage"] == stage]

    def returns(self) -> pd.DataFrame:
        return pd.DataFrame(self._returns)

    def n_trials(self, stage: str | None = None) -> int:
        """Distinct configurations attempted (failed ones count too)."""
        return len({r["trial_id"] for r in self.records(stage)})

    def trial_sharpes(self) -> np.ndarray:
        R = self.returns()
        if R.empty:
            return np.array([])
        return (R.mean() / R.std(ddof=1)).to_numpy()


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
            out[k] = v
    return out

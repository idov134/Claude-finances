"""One-shot holdout: the last >= 24 months are physically removed during research and opened once."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..errors import HoldoutError


def holdout_start(dates: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    if months < 24:
        raise HoldoutError("holdout must be at least 24 months")
    last = pd.DatetimeIndex(dates)[-1]
    return (last - pd.DateOffset(months=months)).normalize() + pd.Timedelta(days=1)


class HoldoutLock:
    def __init__(self, artifacts: str | Path):
        self.path = Path(artifacts) / "holdout_lock.json"

    @property
    def opened(self) -> bool:
        return self.path.exists()

    def info(self) -> dict | None:
        return json.loads(self.path.read_text()) if self.opened else None

    def open(self, config_hash: str, spec: dict, force: bool = False) -> dict:
        if self.opened and not force:
            info = self.info()
            raise HoldoutError(f"holdout already opened at {info['opened_at']} (config {info['config_hash']}). "
                               "Re-running it turns it into in-sample data. Use force only if you accept "
                               "that the report will be marked CONTAMINATED.")
        prev = self.info()
        rec = {"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": config_hash, "spec": spec,
               "n_opens": (prev or {}).get("n_opens", 0) + 1}
        rec["contaminated"] = rec["n_opens"] > 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=2, default=str))
        return rec


def assert_research_data(bundle, hstart: pd.Timestamp) -> None:
    """Guard used by the research pipeline: no data on/after the holdout start may be present."""
    if bundle.dates[-1] >= hstart:
        raise HoldoutError(f"research data reaches {bundle.dates[-1].date()} >= holdout start {hstart.date()}")

"""Full research trial history across experiments (for DSR/PBO charged on the whole search).

Sources are append-only registries committed under ``docs/``. Distinct trial IDs count toward N
whether they completed, failed, or were planned and never finished — the conservative reading of
"every configuration tried".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# (name, folder, returns file, index convention)
SOURCES = [
    ("original_research_v1", "docs/research_history/original_research_v1", "returns.csv", "decision"),
    ("gpt_reentry_v1", "docs/reentry_review", "returns.csv", "calendar"),
    ("gpt_research_comparison_v1", "docs/research_comparison", "trial_returns.csv", "decision"),
    ("algo_research_v2", "docs/algo_research_v2", "returns.csv", "decision"),
]


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_history(root: str | Path | None = None, extra: dict | None = None) -> dict:
    """Return {'n_trials', 'n_completed', 'by_source', 'returns' (decision-indexed, completed only)}.

    ``extra`` = {"name": (records, returns_frame)} adds an in-progress experiment not yet on disk.
    """
    root = Path(root) if root else REPO
    ids, completed, by_source, frames = set(), set(), {}, []
    sources = [(n, root / f, r, c) for n, f, r, c in SOURCES]
    for name, folder, rfile, conv in sources:
        recs = _records(folder / "registry.jsonl")
        if not recs:
            continue
        sid = {f"{name}:{r['trial_id']}" for r in recs}
        ok = {f"{name}:{r['trial_id']}" for r in recs if r.get("status") == "ok"}
        ids |= sid
        completed |= ok
        by_source[name] = {"distinct": len(sid), "completed": len(ok), "index": conv}
        rp = folder / rfile
        if conv == "decision" and rp.exists():
            df = pd.read_csv(rp, index_col=0, parse_dates=True)
            df.columns = [f"{name}:{c}" for c in df.columns]
            frames.append(df[[c for c in df.columns if c in ok]])
    for name, (recs, df) in (extra or {}).items():
        sid = {f"{name}:{r['trial_id']}" for r in recs}
        ok = {f"{name}:{r['trial_id']}" for r in recs if r.get("status") == "ok"}
        ids |= sid
        completed |= ok
        by_source[name] = {"distinct": len(sid), "completed": len(ok), "index": "decision"}
        if df is not None and len(df.columns):
            d = df.copy()
            d.columns = [f"{name}:{c}" for c in d.columns]
            frames.append(d[[c for c in d.columns if c in ok]])
    returns = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    return {"n_trials": len(ids), "n_completed": len(completed), "by_source": by_source, "returns": returns}


def trial_sharpes(returns: pd.DataFrame) -> np.ndarray:
    if returns.empty:
        return np.array([])
    return (returns.mean() / returns.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

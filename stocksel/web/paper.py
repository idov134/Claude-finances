"""Previewed paper decisions backed by the existing ledger; no brokerage path."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..live.feeds import utc_now
from ..live.paper import PaperLedger


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as fh:
            json.dump(value, fh, allow_nan=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PaperWorkflow:
    def __init__(self, service):
        self.service = service
        self.root = Path(service.cfg.live.ledger_dir) / "previews"

    def _revision(self):
        path = Path(self.service.cfg.live.ledger_dir) / "ledger.json"
        return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()

    def _ready(self):
        s = self.service
        if s.status != "ready" or s.bundle is None:
            raise ValueError("Wait for the research scan to finish")
        if s.feed_name != "synthetic" or s.bundle.meta.get("source") != "synthetic":
            raise ValueError("Web paper execution currently supports the synthetic demo only; real-data corporate actions are not reconciled")

    def preview(self, ticker, position_pct, thesis="", counter_thesis=""):
        self._ready()
        s = self.service
        row = next((r for r in s.rows if r["ticker"] == ticker), None)
        quote = s.quotes.get(ticker)
        if row is None or quote is None:
            raise ValueError("Ticker must have an eligible signal and a quote")
        if not quote.simulated or not quote.connected or (utc_now() - quote.received_at).total_seconds() > 120:
            raise ValueError("Quote is unavailable, disconnected or stale; refresh before previewing")
        if not 0 <= position_pct <= 100 * s.cfg.portfolio.max_position:
            raise ValueError("Target exceeds the configured per-position risk cap")
        asof = quote.data_timestamp.date().isoformat()
        if not (s.bundle.dates > pd.Timestamp(asof)).any():
            raise ValueError("Replay ended; no next session remains")
        ledger = PaperLedger(s.cfg)
        if ledger.state.get("pending"):
            raise ValueError("Advance the pending paper order before creating another decision")
        if ledger.state.get("last_update") and ledger.state["last_update"] > asof:
            raise ValueError("Replay is behind the saved ledger; resume its recorded session")
        prices = s.bundle.close.loc[pd.Timestamp(asof)]
        current = ledger.weights(prices)
        weights = current.to_dict()
        weights[ticker] = position_pct / 100
        if sum(weights.values()) > s.cfg.portfolio.max_gross + 1e-8:
            raise ValueError("Target would exceed available unleveraged capital")
        sector = s.bundle.sectors.get(ticker)
        sector_weight = sum(w for t, w in weights.items() if s.bundle.sectors.get(t) == sector)
        if sector_weight > s.cfg.portfolio.max_sector + 1e-8:
            raise ValueError("Target would exceed the configured sector cap")
        equity = float(ledger.state["cash"]) + sum(float(p["shares"]) * float(prices.get(t, p["last"]))
                                                  for t, p in ledger.state["positions"].items())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError("Portfolio equity cannot be valued at this replay date")
        value = (position_pct / 100 - current.get(ticker, 0)) * equity
        adv = float(s.bundle.adv_usd(63).loc[pd.Timestamp(asof), ticker])
        cost = float(CostModel(s.cfg.costs).rate([abs(value)], [adv], [.02])[0]) * abs(value)
        pid = uuid.uuid4().hex
        preview = {"preview_id": pid, "asof": asof, "ticker": ticker,
                   "current_weight_pct": float(current.get(ticker, 0) * 100), "target_weight_pct": position_pct,
                   "estimated_trade_value": value, "estimated_cost": cost, "equity": equity,
                   "currency": s.cfg.capital.currency, "execution": "next available open",
                   "expires_at": (utc_now() + timedelta(minutes=5)).isoformat(),
                   "warnings": ["Synthetic paper decision; does not establish a profitable strategy.",
                                "Cost estimate uses current liquidity and assumed 2% daily volatility; fills and costs depend on the next open."],
                   "config_hash": s.cfg.hash(), "revision": self._revision(), "weights": weights,
                   "stops": {t: (row.get("stop") if t == ticker else p.get("stop"))
                             for t, p in {**ledger.state["positions"], ticker: {}}.items()},
                   "journal": {"ticker": ticker, "thesis": thesis, "counter_thesis": counter_thesis,
                               "signal_asof": row["signal_asof"], "horizon_months": s.cfg.labels.horizon_months,
                               "calibrated_probability": row.get("calibrated_prob"), "confidence": row.get("confidence")}}
        atomic_json(self.root / (pid + ".json"), preview)
        return {k: v for k, v in preview.items() if k not in ("weights", "stops", "revision")}

    def submit(self, preview_id, idempotency_key):
        self._ready()
        path = self.root / (preview_id + ".json")
        if not path.exists():
            raise ValueError("Preview not found; create a fresh preview")
        preview = json.loads(path.read_text())
        existing = PaperLedger(self.service.cfg).submission(idempotency_key)
        if existing:
            if existing.get("journal", {}).get("preview_id") != preview_id:
                raise ValueError("Idempotency key was already used for a different preview")
            return {"order_id": existing["order_id"], "status": existing["status"], "duplicate": True}
        receipt = self.root / ("receipt-" + hashlib.sha256(idempotency_key.encode()).hexdigest() + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["preview_id"] != preview_id:
                raise ValueError("Idempotency key was already used for a different preview")
            return {**saved["result"], "duplicate": True}
        if pd.Timestamp(preview["expires_at"]) < pd.Timestamp(utc_now()):
            raise ValueError("Preview expired; review a fresh estimate")
        if preview["config_hash"] != self.service.cfg.hash() or preview["revision"] != self._revision():
            raise ValueError("Portfolio or configuration changed since preview; review a new estimate")
        quote = self.service.quotes.get(preview["ticker"])
        if quote is None or not quote.connected or (utc_now() - quote.received_at).total_seconds() > 120 or quote.data_timestamp.date().isoformat() != preview["asof"]:
            raise ValueError("Replay moved since preview; review a new estimate")
        recs = pd.DataFrame([{"ticker": t, "position_pct": w * 100, "stop": preview["stops"].get(t)}
                             for t, w in preview["weights"].items()])
        result = PaperLedger(self.service.cfg).submit(recs, preview["asof"],
                    idempotency_key=idempotency_key, journal={**preview["journal"], "preview_id": preview_id})
        atomic_json(receipt, {"preview_id": preview_id, "result": result})
        return result

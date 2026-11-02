"""Paper-trading ledger (3-6 months required before real money).

Orders are created from recommendations at a close and filled at the next
available open from the data feed; stops are checked at closes. State is a
JSON file so the ledger survives restarts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel


class PaperLedger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.live.ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "start": None, "cash": float(cfg.capital.initial), "positions": {}, "pending": [], "equity": [],
            "trades": [], "last_update": None}

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=2, default=str))

    def weights(self, prices: pd.Series) -> pd.Series:
        pos = self.state["positions"]
        vals = {t: p["shares"] * float(prices.get(t, p["last"])) for t, p in pos.items()}
        eq = self.state["cash"] + sum(vals.values())
        return pd.Series({t: v / eq for t, v in vals.items()}, dtype=float)

    def submit(self, recs: pd.DataFrame, asof: str):
        """Queue target weights; filled at the next open by ``update``."""
        if self.state["start"] is None:
            self.state["start"] = asof
        targets = {r.ticker: r.position_pct / 100 for r in recs.itertuples()}
        stops = {r.ticker: r.stop for r in recs.itertuples()}
        self.state["pending"].append({"asof": asof, "targets": targets, "stops": stops})
        self.save()

    def update(self, bundle):
        cm = CostModel(self.cfg.costs)
        last = pd.Timestamp(self.state["last_update"]) if self.state["last_update"] else None
        dates = bundle.dates if last is None else bundle.dates[bundle.dates > last]
        adv = bundle.adv_usd(63)
        for d in dates:
            o, c = bundle.open.loc[d], bundle.close.loc[d]
            # fills at the open
            todo = [p for p in self.state["pending"] if pd.Timestamp(p["asof"]) < d]
            for order in todo:
                eq = self.state["cash"] + sum(p["shares"] * float(o.get(t, np.nan) if np.isfinite(o.get(t, np.nan)) else p["last"])
                                              for t, p in self.state["positions"].items())
                tickers = set(order["targets"]) | set(self.state["positions"])
                for t in sorted(tickers, key=lambda x: order["targets"].get(x, 0)):  # sells first
                    px = float(o.get(t, np.nan))
                    if not np.isfinite(px):
                        continue
                    cur = self.state["positions"].get(t, {"shares": 0.0})["shares"]
                    tgt = order["targets"].get(t, 0.0) * eq / px
                    dsh = tgt - cur
                    if abs(dsh * px) < 1:
                        continue
                    cost = float(cm.rate(np.array([abs(dsh * px)]), np.array([adv.loc[d].get(t, 1e7)]), np.array([0.02]))[0]
                                 * abs(dsh * px))
                    self.state["cash"] -= dsh * px + cost
                    p = self.state["positions"].setdefault(t, {"shares": 0.0, "entry": str(d.date()), "entry_px": px,
                                                               "last": px, "stop": order["stops"].get(t)})
                    p["shares"] += dsh
                    self.state["trades"].append({"date": str(d.date()), "ticker": t, "shares": dsh, "px": px, "cost": cost})
                    if abs(p["shares"]) < 1e-9:
                        self.state["positions"].pop(t)
                self.state["pending"].remove(order)
            # stops at the close -> exit next open
            for t, p in list(self.state["positions"].items()):
                px = float(c.get(t, np.nan))
                if np.isfinite(px):
                    p["last"] = px
                    if p.get("stop") is not None and px < p["stop"]:
                        self.state["pending"].append({"asof": str(d.date()), "targets": {
                            k: v for k, v in self.weights(c).items() if k != t}, "stops": {}})
            eq = self.state["cash"] + sum(p["shares"] * p["last"] for p in self.state["positions"].values())
            self.state["equity"].append({"date": str(d.date()), "equity": eq,
                                         "bench": float(bundle.benchmark["close"].loc[d])})
            self.state["last_update"] = str(d.date())
        self.save()

    def equity(self) -> pd.DataFrame:
        if not self.state["equity"]:
            return pd.DataFrame(columns=["equity", "bench"])
        df = pd.DataFrame(self.state["equity"]).drop_duplicates("date", keep="last")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def status(self) -> dict:
        eq = self.equity()
        months = 0.0 if not self.state["start"] or eq.empty else (eq.index[-1] - pd.Timestamp(self.state["start"])).days / 30.44
        lv = self.cfg.live
        return {"start": self.state["start"], "months": round(months, 1), "live_ready": months >= lv.paper_min_months,
                "recommended_ready": months >= lv.paper_recommended_months,
                "equity": float(eq["equity"].iloc[-1]) if len(eq) else float(self.cfg.capital.initial),
                "n_positions": len(self.state["positions"])}

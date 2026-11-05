"""Atomic, idempotent paper accounting on one fixed adjusted-price basis.

This ledger models fractional units of a canonical adjusted-price series, not
brokerage share counts. Raw quotes, explicit cash dividends/splits and external
cash flows are not supported. Dividend effects already in adjusted prices must
not be credited a second time. Orders execute only at a later available open.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class PaperLedger:
    SCHEMA_VERSION = 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.live.ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.json"
        self.backup_path = self.dir / "ledger.json.bak"
        with _LOCKS_GUARD:
            self._thread_lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())
        self.state = self._load()
        self._revision = self.state["revision"]

    def _new(self):
        if not _finite(self.cfg.capital.initial) or float(self.cfg.capital.initial) <= 0:
            raise ValueError("Initial paper capital must be positive and finite")
        return {"schema_version": self.SCHEMA_VERSION, "revision": 0,
                "initial_capital": float(self.cfg.capital.initial), "currency": self.cfg.capital.currency,
                "price_basis": "canonical_adjusted", "price_source": None,
                "start": None, "cash": float(self.cfg.capital.initial), "positions": {},
                "pending": [], "orders": [], "journal": [], "equity": [], "trades": [],
                "fees_paid": 0., "realized_pnl": 0., "dividend_income": None,
                "external_cash_flows_supported": False, "accounting_warnings": [], "last_update": None}

    def _load(self):
        if not self.path.exists():
            return self._new()
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Paper ledger unreadable; preserve it and inspect ledger.json.bak for explicit recovery") from exc
        if state.get("schema_version", 1) > self.SCHEMA_VERSION:
            raise ValueError("Paper ledger schema is newer than this application")
        if state.get("schema_version", 1) == 1:
            state = self._migrate(state)
        if state.get("currency") != self.cfg.capital.currency:
            raise ValueError("Ledger currency differs from config; currency conversion is not supported")
        return state

    def _migrate(self, old):
        """Preserve legacy records; reconstruct basis only when the trade log supports it."""
        state = {**self._new(), **copy.deepcopy(old), "schema_version": 2}
        trades = state["trades"]
        reconstructed = {}
        realized, fees = 0., 0.
        for t in trades:
            symbol, qty, px, fee = t["ticker"], float(t["shares"]), float(t["px"]), float(t.get("cost", 0))
            p = reconstructed.setdefault(symbol, {"shares": 0., "basis": 0.})
            fees += fee
            if qty > 0:
                p["shares"] += qty
                p["basis"] += qty * px + fee
            elif p["shares"] >= -qty and p["shares"] > 0:
                released = p["basis"] * (-qty / p["shares"])
                if realized is not None:
                    realized += -qty * px - fee - released
                p["shares"] += qty
                p["basis"] -= released
            else:
                realized = None
        complete = realized is not None
        for symbol, p in state["positions"].items():
            r = reconstructed.get(symbol, {})
            known = bool(r and math.isclose(r["shares"], p["shares"], abs_tol=1e-6))
            p["cost_basis"] = r["basis"] if known else None
            p["basis_known"] = known
            p["last_date"] = state.get("last_update")
            complete &= known
        state["fees_paid"] = fees if complete else None
        state["realized_pnl"] = realized if complete else None
        state["initial_capital"] = (float(state["cash"]) + sum(float(t["shares"]) * float(t["px"]) + float(t.get("cost", 0)) for t in trades)) if complete else None
        if not complete:
            state["accounting_warnings"].append("Legacy trade history cannot establish initial capital/cost basis; unavailable P&L components remain null")
        for i, order in enumerate(state["pending"]):
            order.setdefault("order_id", "legacy-" + str(i) + "-" + _digest(order)[:16])
            order.setdefault("idempotency_key", order["order_id"])
            order.setdefault("payload_hash", _digest({"asof": order["asof"], "targets": order["targets"]}))
            order.setdefault("kind", "rebalance")
            order.setdefault("status", "pending")
            state["orders"].append(copy.deepcopy(order))
        state["accounting_warnings"].append("Migrated legacy ledger; original bytes retained in ledger.json.bak on the first write")
        return state

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            with (self.dir / "ledger.lock").open("a+b") as fh:
                if os.name == "nt":  # Windows local startup
                    import msvcrt
                    fh.seek(0); fh.write(b"0"); fh.flush(); fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if os.name == "nt":
                        fh.seek(0); msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _atomic(self, path, content):
        fd, tmp = tempfile.mkstemp(prefix=".ledger-", dir=self.dir)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(content); fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _persist(self):
        # Serialize first: NaN or nonserializable input cannot destroy the prior file.
        self.state["revision"] += 1
        text = json.dumps(self.state, indent=2, allow_nan=False, default=str)
        if self.path.exists():
            prior = self.path.read_text()
            json.loads(prior)
            self._atomic(self.backup_path, prior)
        self._atomic(self.path, text)
        self._revision = self.state["revision"]

    def save(self):
        """Explicit save supports callers, refusing stale-object overwrites."""
        with self._locked():
            if self.path.exists() and self._load()["revision"] != self._revision:
                raise ValueError("Ledger changed concurrently; reload before saving")
            self._persist()

    def weights(self, prices: pd.Series) -> pd.Series:
        vals = {t: p["shares"] * (float(prices.get(t)) if _finite(prices.get(t)) else p["last"])
                for t, p in self.state["positions"].items()}
        eq = self.state["cash"] + sum(vals.values())
        return pd.Series({t: v / eq for t, v in vals.items()}, dtype=float) if eq > 0 else pd.Series(dtype=float)

    def submission(self, idempotency_key: str):
        """Recover a receipt after a response/file-write interruption."""
        with self._locked():
            self.state = self._load()
            return next((copy.deepcopy(order) for order in self.state["orders"]
                         if order["idempotency_key"] == idempotency_key), None)

    def submit(self, recs: pd.DataFrame, asof: str, idempotency_key: str | None = None, journal: dict | None = None):
        """Queue a full target portfolio; omitted holdings have target zero.

        The optional idempotency key survives restarts. Reuse with a different
        payload fails; retries of the same payload return the original receipt.
        """
        date = pd.Timestamp(asof)
        if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
            raise ValueError("asof must be a date without a timezone or time")
        asof = str(date.date())
        targets, stops = {}, {}
        for row in recs.itertuples():
            ticker = str(row.ticker)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-^=]{0,31}", ticker) or ticker in targets:
                raise ValueError("Tickers must be unique, valid identifiers")
            weight = float(row.position_pct)
            if not math.isfinite(weight) or not 0 <= weight <= 100:
                raise ValueError("Position percentages must be finite values between 0 and 100")
            targets[ticker] = weight / 100
            stop = getattr(row, "stop", None)
            if stop is not None and not pd.isna(stop):
                if not _finite(stop) or float(stop) <= 0:
                    raise ValueError("Stops must be positive finite prices or null")
                stop = float(stop)
            else:
                stop = None
            stops[ticker] = stop
        if sum(targets.values()) > min(1., float(self.cfg.portfolio.max_gross)) + 1e-9:
            raise ValueError("Target weights exceed the long-only exposure limit")
        journal = journal or {}
        if not isinstance(journal, dict) or len(json.dumps(journal, allow_nan=False)) > 16000:
            raise ValueError("Journal must be a JSON object no larger than 16 KB")
        payload = {"asof": asof, "targets": targets, "stops": stops, "journal": journal}
        fingerprint = _digest(payload)
        key = idempotency_key or fingerprint
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError("Idempotency key must contain 1 to 128 characters")
        with self._locked():
            self.state = self._load()
            for order in self.state["orders"]:
                if order["idempotency_key"] == key:
                    if order["payload_hash"] != fingerprint:
                        raise ValueError("Idempotency key already belongs to a different decision")
                    return {"order_id": order["order_id"], "duplicate": True, "status": order["status"]}
            if self.state["last_update"] and date < pd.Timestamp(self.state["last_update"]):
                raise ValueError("Cannot submit a decision before the ledger's latest processed date")
            if self.state["pending"]:
                raise ValueError("A paper order is already pending; process it before another decision")
            order = {**payload, "order_id": fingerprint[:24], "idempotency_key": key, "payload_hash": fingerprint,
                     "kind": "rebalance", "status": "pending"}
            if self.state["start"] is None:
                self.state["start"] = asof
            self.state["pending"].append(order)
            self.state["orders"].append(copy.deepcopy(order))
            self.state["journal"].append({"order_id": order["order_id"], "asof": asof, **journal})
            self._persist()
            return {"order_id": order["order_id"], "duplicate": False, "status": "pending"}

    def _sync_order(self, order):
        for i, record in enumerate(self.state["orders"]):
            if record["order_id"] == order["order_id"]:
                self.state["orders"][i] = copy.deepcopy(order)
                return
        self.state["orders"].append(copy.deepcopy(order))

    def _trade(self, ticker, delta, px, cost, date, order, estimated):
        positions = self.state["positions"]
        p = positions.get(ticker)
        realized = None
        if delta > 0:
            if p is None:
                p = positions[ticker] = {"shares": 0., "entry": date, "entry_px": px, "last": px,
                                          "stop": None, "cost_basis": 0., "basis_known": True}
            old_shares = p["shares"]
            p["entry_px"] = (old_shares * p["entry_px"] + delta * px) / (old_shares + delta)
            if p.get("cost_basis") is not None:
                p["cost_basis"] += delta * px + cost
            p["shares"] += delta
        else:
            released = p["cost_basis"] * (-delta / p["shares"]) if p.get("cost_basis") is not None else None
            realized = -delta * px - cost - released if released is not None else None
            if self.state["realized_pnl"] is not None:
                self.state["realized_pnl"] = self.state["realized_pnl"] + realized if realized is not None else None
            if released is not None:
                p["cost_basis"] -= released
            p["shares"] += delta
        self.state["cash"] -= delta * px + cost
        if self.state["fees_paid"] is not None:
            self.state["fees_paid"] += cost
        p["last"] = px
        if ticker in order["stops"]:
            p["stop"] = order["stops"][ticker]
        self.state["trades"].append({"date": date, "ticker": ticker, "shares": delta, "px": px,
                                      "cost": cost, "realized_pnl": realized, "order_id": order["order_id"],
                                      "cost_inputs_estimated": estimated})
        if p["shares"] < 1e-9:
            positions.pop(ticker)

    def _settle_delistings(self, bundle, d):
        """Consume explicit synthetic settlements once; real event dates are required.

        Synthetic delist_returns become known after the last valid price and
        settle on the next represented session. A missing quote by itself is
        never a delisting event. Real adapters currently provide no dated cash
        settlement contract, so those cases require reconciliation.
        """
        date = str(d.date())
        for ticker, p in list(self.state["positions"].items()):
            if ticker not in bundle.delist_returns.index or ticker not in bundle.tickers:
                continue
            if _finite(bundle.close.at[d, ticker]):
                continue
            last_valid = bundle.close[ticker].last_valid_index()
            if last_valid is None or last_valid >= d:
                continue
            if bundle.meta.get("source") != "synthetic":
                raise ValueError("Known delisting needs dated real-provider settlement reconciliation before ledger advance")
            rate = bundle.delist_returns[ticker]
            if not _finite(rate) or float(rate) < -1:
                raise ValueError("Synthetic delisting return must be finite and at least -100%")
            price = float(bundle.close.at[last_valid, ticker]) * (1 + float(rate))
            oid = "delisting-" + ticker + "-" + date
            order = {"order_id": oid, "idempotency_key": oid, "payload_hash": oid,
                     "kind": "delisting", "status": "settled", "asof": date,
                     "targets": {}, "stops": {}, "synthetic_settlement": True}
            self._trade(ticker, -p["shares"], price, 0., date, order, False)
            self._sync_order(order)
            for pending in list(self.state["pending"]):
                if ticker not in pending.get("targets", {}) and ticker not in pending.get("remaining", []):
                    continue
                pending["targets"].pop(ticker, None)
                pending["stops"].pop(ticker, None)
                pending.get("target_notional", {}).pop(ticker, None)
                for key in ("scope", "remaining"):
                    if ticker in pending.get(key, []):
                        pending[key].remove(ticker)
                pending.setdefault("warnings", []).append(f"{ticker} resolved by explicit synthetic delisting settlement")
                if "remaining" in pending and not pending["remaining"]:
                    pending["status"] = "filled"
                    self.state["pending"].remove(pending)
                self._sync_order(pending)

    def update(self, bundle):
        """Process completed historical sessions; repeat calls are idempotent."""
        bundle.validate()
        basis = bundle.meta.get("price_basis", "canonical_adjusted")
        if basis not in ("canonical_adjusted", "adjusted"):
            raise ValueError("Paper ledger requires canonical adjusted prices; raw corporate actions are not reconciled")
        with self._locked():
            self.state = self._load()
            if not self.state["start"]:
                return
            if bundle.meta.get("currency", "USD") != self.state["currency"]:
                raise ValueError("Data currency differs from ledger currency; dated FX conversion is not supported")
            source = bundle.meta.get("source", "unspecified")
            if self.state["price_source"] not in (None, source):
                raise ValueError("Cannot mix market-data sources inside an existing paper account")
            self.state["price_source"] = source
            for ticker, p in self.state["positions"].items():
                mark = p.get("last_date")
                if mark and pd.Timestamp(mark) in bundle.dates and ticker in bundle.tickers:
                    px = bundle.close.at[pd.Timestamp(mark), ticker]
                    if _finite(px) and not math.isclose(float(px), p["last"], rel_tol=1e-7, abs_tol=1e-8):
                        raise ValueError("Historical marks changed; reconcile adjusted share basis before advancing the ledger")
            last = pd.Timestamp(self.state["last_update"]) if self.state["last_update"] else None
            dates = bundle.dates[bundle.dates > last] if last is not None else bundle.dates[bundle.dates >= pd.Timestamp(self.state["start"])]
            if not len(dates):
                return
            cm = CostModel(self.cfg.costs)
            adv = bundle.adv_usd(63).shift(1)
            vol = bundle.close.pct_change(fill_method=None).rolling(63, min_periods=5).std().shift(1)
            for d in dates:
                date = str(d.date())
                o, c = bundle.open.loc[d], bundle.close.loc[d]
                for order in list(self.state["pending"]):
                    if pd.Timestamp(order["asof"]) >= d:
                        continue
                    if "remaining" not in order:
                        eq = self.state["cash"] + sum(p["shares"] * (float(o.get(t)) if _finite(o.get(t)) else p["last"])
                                                      for t, p in self.state["positions"].items())
                        scope = order.get("scope", sorted(set(order["targets"]) | set(self.state["positions"])))
                        order["remaining"] = list(scope)
                        order["target_notional"] = {t: order["targets"].get(t, 0.) * eq for t in scope}
                    executable = []
                    for ticker in order["remaining"]:
                        px = o.get(ticker)
                        if not _finite(px) or float(px) <= 0:
                            continue
                        px = float(px)
                        cur = self.state["positions"].get(ticker, {}).get("shares", 0.)
                        delta = order["target_notional"][ticker] / px - cur
                        executable.append((delta * px, ticker, delta, px))
                    # Actual reductions are executed before purchases, independent of target rank.
                    priced = {item[1] for item in executable}
                    waiting_for_sale = any(t not in priced and
                        order["target_notional"][t] < self.state["positions"].get(t, {}).get("shares", 0.) *
                        self.state["positions"].get(t, {}).get("last", 0.) for t in order["remaining"])
                    for _, ticker, delta, px in sorted(executable):
                        av = adv.loc[d].get(ticker, np.nan)
                        vv = vol.loc[d].get(ticker, np.nan)
                        estimated = not (_finite(av) and _finite(vv))
                        def charge(notional):
                            return float(cm.rate([notional], [av], [vv])[0] * abs(notional))
                        if delta > 0:
                            desired = delta * px
                            cash = max(0., self.state["cash"])
                            if desired + charge(desired) > cash:
                                if waiting_for_sale:
                                    continue  # retain the buy until the missing sale can fund it
                                lo, hi = 0., min(desired, cash)
                                for _ in range(45):
                                    mid = (lo + hi) / 2
                                    if mid + charge(mid) <= cash:
                                        lo = mid
                                    else:
                                        hi = mid
                                delta = lo / px
                                order.setdefault("warnings", [])
                                if "Purchases capped to available cash including fees" not in order["warnings"]:
                                    order["warnings"].append("Purchases capped to available cash including fees")
                        if abs(delta * px) > 1e-8:
                            self._trade(ticker, delta, px, charge(abs(delta * px)), date, order, estimated)
                        order["remaining"].remove(ticker)
                    order["status"] = "partial" if order["remaining"] else "filled"
                    self._sync_order(order)
                    if not order["remaining"]:
                        self.state["pending"].remove(order)
                self._settle_delistings(bundle, d)
                # A stop liquidates only its named holding; other target weights remain untouched.
                for ticker, p in list(self.state["positions"].items()):
                    px = c.get(ticker)
                    if not _finite(px) or float(px) <= 0:
                        continue
                    p["last"], p["last_date"] = float(px), date
                    if p.get("stop") is not None and float(px) < p["stop"]:
                        duplicate = any(v.get("kind") == "stop" and ticker in v.get("scope", []) for v in self.state["pending"])
                        if not duplicate:
                            oid = "stop-" + ticker + "-" + date
                            order = {"order_id": oid, "idempotency_key": oid, "payload_hash": oid, "kind": "stop",
                                     "asof": date, "targets": {ticker: 0.}, "stops": {}, "scope": [ticker], "status": "pending"}
                            self.state["pending"].append(order)
                            self._sync_order(order)
                eq = self.state["cash"] + sum(p["shares"] * p["last"] for p in self.state["positions"].values())
                bench = bundle.benchmark["close"].loc[d]
                self.state["equity"].append({"date": date, "equity": eq,
                                             "bench": float(bench) if _finite(bench) else None})
                self.state["last_update"] = date
            self._persist()

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
        value = self.state["cash"] + sum(p["shares"] * p["last"] for p in self.state["positions"].values())
        known = all(p.get("cost_basis") is not None for p in self.state["positions"].values())
        unrealized = sum(p["shares"] * p["last"] - p["cost_basis"] for p in self.state["positions"].values()) if known else None
        initial = self.state["initial_capital"]
        return {"start": self.state["start"], "months": round(months, 1), "live_ready": False,
                "paper_duration_met": months >= lv.paper_min_months,
                "recommended_ready": False, "recommended_duration_met": months >= lv.paper_recommended_months,
                "equity": float(value), "cash": self.state["cash"], "initial_capital": initial,
                "cumulative_return": value / initial - 1 if initial else None,
                "time_weighted_return": value / initial - 1 if initial else None,
                "money_weighted_return": None, "external_cash_flows_supported": False,
                "realized_pnl": self.state["realized_pnl"], "unrealized_pnl": unrealized,
                "fees_paid": self.state["fees_paid"], "dividend_income": None,
                "currency": self.state["currency"], "price_basis": self.state["price_basis"],
                "n_positions": len(self.state["positions"]), "pending_orders": len(self.state["pending"]),
                "accounting_warnings": self.state["accounting_warnings"],
                "readiness_note": "Elapsed paper time alone does not establish a real-market advantage"}

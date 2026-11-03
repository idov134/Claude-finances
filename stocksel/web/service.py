"""Background research scanner; quote updates never imply new model evidence."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import vol_target
from ..data import load_bundle
from ..live.feeds import global_freshness, make_feed, market_status, utc_now
from ..live.recommend import recommend
from ..validation.holdout import holdout_start
from ..validation.readiness import system_validated
from ..validation.trials import TrialRegistry

log = logging.getLogger(__name__)
DISCLAIMER = "Research and paper trading only. Synthetic results do not establish real-market profits. Confidence is capped at Low until real-data validation passes."


class SetupError(ValueError):
    """A local, credential-free message safe to display in the dashboard."""


def clean(value):
    """Strict JSON, including unavailable/numpy values from research outputs."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MarketService:
    def __init__(self, cfg, feed_name=None, bundle=None, feed=None, scanner=None):
        self.cfg = cfg
        self.feed_name = feed_name or cfg.live.get("feed", "synthetic")
        self.bundle, self.feed = bundle, feed
        self.scanner = scanner or recommend
        self.rows, self.meta, self.quotes = [], {}, {}
        self.spec, self.alerts = {}, {}
        self.status, self.error, self.feed_error = "loading", None, None
        self.scored_at = None
        self._tasks = []
        self._history_loaded = 0.0
        self._last_poll = None
        self._next_scan_at = None
        self._portfolio = {"positions": [], "history": []}
        self._position_tickers = set()
        self._high_water = {}

    async def start(self):
        if not self._tasks:
            self._tasks = [asyncio.create_task(self._run(), name="stocksel-market-worker")]

    async def close(self):
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self.feed is not None:
            await self.feed.close()

    def _alert(self, key, message, level="warning", ticker=None):
        self.alerts[key] = {"id": key, "message": message, "level": level,
                            "ticker": ticker, "time": utc_now().isoformat()}
        # Keyed current conditions, bounded in long-running scans.
        if len(self.alerts) > 300:
            self.alerts.pop(next(iter(self.alerts)))

    def _bootstrap(self):
        if self.bundle is None:
            self.bundle = load_bundle(self.cfg)
        if self.feed_name != "synthetic" and self.bundle.meta.get("source") == "synthetic":
            raise SetupError("Real quote feeds require real historical data: configure data.provider (e.g. csv/sharadar). Synthetic S0000 symbols cannot be sent to a real exchange.")
        if self.feed_name == "synthetic":
            # Do not evaluate or reuse the demo's already-opened final holdout.
            boundary = holdout_start(self.bundle.dates, int(self.cfg.validation.holdout_months))
            self.bundle = self.bundle.truncate(boundary - pd.Timedelta(days=1))
        self._history_loaded = time.monotonic()
        self._scan(force=True)
        self._portfolio = self._read_portfolio()

    def _decision_date(self):
        # Reject the final partial month even if a provider ends early/stops updating.
        dates = self.bundle.month_end_dates()
        last = self.bundle.dates[-1]
        if self.feed_name == "synthetic" or (last + pd.offsets.BDay(1)).month == last.month:
            # The final historical month is replay-only in the demo: monthly
            # decisions must precede every replayed quote, even at a full month-end.
            dates = dates[:-1]
        if len(dates) < 24:
            raise SetupError("At least two years of historical prices are required for monthly signals")
        return dates[-1]

    def _scan(self, force=False):
        decision = self._decision_date()
        asof = str(decision.date())
        if not force and self.meta.get("asof") == asof:
            return
        path = Path(self.cfg.paths.artifacts) / "final_spec.json"
        spec = json.loads(path.read_text()) if path.exists() else None
        if spec and spec.get("config_hash") != self.cfg.hash():
            raise SetupError("Frozen research spec does not match config.yaml. Use the matching config or a separate artifacts path; no silently changed model will be served.")
        provisional = spec is None
        if provisional:
            spec = {"members": [self.cfg.models.baseline], "horizon": self.cfg.labels.horizon_months,
                    "features": list(self.cfg.features.sets[self.cfg.models.feature_set]),
                    "keep": 1 - self.cfg.signals.haircut, "vol_target": vol_target(self.cfg),
                    "dsr": None, "pbo": None, "holdout_passed": False,
                    "config_hash": self.cfg.hash(), "provisional": True}
        registry = TrialRegistry(Path(self.cfg.paths.artifacts) / "trials", stage="dashboard")
        trial = {"purpose": "unvalidated_dashboard_fit" if provisional else "frozen_spec_dashboard_fit",
                 "config_hash": self.cfg.hash(), "asof": asof, "members": spec["members"]}
        try:
            recs, meta = self.scanner(self.cfg, asof=asof, bundle=self.bundle,
                                      spec=spec, include_all=True)
            registry.log(trial, metrics={"n_candidates": meta["n_candidates"]},
                         tags={"performance_evaluation": False, "synthetic": self.feed_name == "synthetic"})
        except Exception as exc:
            registry.log(trial, status="failed", error=type(exc).__name__)
            raise
        previous = {r["ticker"]: r["rank"] for r in self.rows}
        rows = recs.to_dict(orient="records")
        session = market_status(datetime.combine(decision.date(), datetime.min.time(), timezone.utc).replace(hour=17))
        stamp = session.get("session_close") or pd.Timestamp(decision).tz_localize("America/New_York").replace(hour=16).isoformat()
        for rank, row in enumerate(rows, 1):
            row.update(rank=rank, previous_rank=previous.get(row["ticker"]),
                       signal_asof=asof, signal_timestamp=stamp, selected=row["position_pct"] > 0)
            if previous.get(row["ticker"]) not in (None, rank):
                self._alert("rank:" + row["ticker"], f"Monthly rank changed from {previous[row['ticker']]} to {rank}", ticker=row["ticker"])
        self.rows, self.meta, self.spec = rows, meta, spec
        self.scored_at = utc_now().isoformat()
        if provisional:
            self._alert("unvalidated", "No frozen research spec: displaying an unvalidated baseline fit, not a promoted ensemble.")
        if self.feed_name == "synthetic":
            self._alert("simulation", "Synthetic replay only. Prices and research signals are historical simulations; the used holdout is excluded.")
        self._alert("news", "Daily news trading is disabled: the documented net-cost validation failed. No live news stream is connected.", "info")

    def _read_portfolio(self):
        initial = float(self.cfg.capital.initial)
        path = Path(self.cfg.live.ledger_dir) / "ledger.json"
        state = json.loads(path.read_text()) if path.exists() else {"cash": initial, "positions": {}, "equity": []}
        self._position_tickers = set(state.get("positions", {}))
        positions, history = [], []
        for ticker, pos in state.get("positions", {}).items():
            quote = self.quotes.get(ticker)
            # Avoid marking current holdings with an earlier synthetic replay event.
            mark_date = state.get("last_update") or pos["entry"]
            mark_time = (pd.Timestamp(mark_date).normalize().tz_localize("America/New_York")
                         + pd.Timedelta(hours=16)).tz_convert("UTC").to_pydatetime()
            eligible = quote is not None and quote.data_timestamp >= mark_time and self._prices_comparable(quote)
            px = quote.price if eligible else float(pos["last"])
            value = float(pos["shares"]) * px
            positions.append({"ticker": ticker, **pos, "price": px, "value": value,
                              "unrealized_pnl": value - float(pos["shares"]) * float(pos["entry_px"]),
                              "valuation_source": quote.provider if eligible else "ledger last mark",
                              "valuation_timestamp": quote.data_timestamp if eligible else mark_time,
                              "freshness": quote.to_dict()["freshness"] if eligible else {"state": "amber", "label": "Ledger historical mark", "icon": "◷"}})
        equity = float(state["cash"]) + sum(p["value"] for p in positions)
        raw = state.get("equity", [])
        # Anchor both curves at the decision-date close (before next-open fills),
        # not at the first post-trade close, which omits first-day benchmark P&L.
        start = state.get("start")
        reference = self.bundle.benchmark.loc[:start, "close"] if start and self.bundle is not None else pd.Series(dtype=float)
        base_bench = float(reference.iloc[-1]) if len(reference) else None
        for row in raw:
            history.append({"date": row["date"], "equity": row["equity"],
                            "benchmark": initial * row["bench"] / base_bench if base_bench else None})
        peak = max([initial, equity] + [r["equity"] for r in raw])
        return {"initial_capital": initial, "equity": equity, "cash": float(state["cash"]),
                "pnl": equity - initial, "pnl_pct": equity / initial - 1,
                "drawdown": equity / peak - 1, "positions": positions, "history": history,
                "benchmark_return": history[-1]["benchmark"] / initial - 1 if history and base_bench else None,
                "last_update": state.get("last_update"), "read_only": True}

    def _prices_comparable(self, quote):
        # Historical panels/ledger shares are adjusted. Raw live trades must not
        # be compared across an unverified corporate-action basis.
        return quote.simulated or (self.bundle is not None and
            self.bundle.meta.get("raw_quote_basis_asof") == quote.data_timestamp.date().isoformat())

    def _check_risk(self):
        held = {p["ticker"]: p for p in self._portfolio.get("positions", [])}
        for ticker, pos in held.items():
            q = self.quotes.get(ticker)
            if q is None:
                continue
            if not self._prices_comparable(q):
                self._alert("price-basis", "Raw real-time quotes are not reconciled with adjusted historical shares/corporate actions. Live stop comparisons and quote-based ledger marks are withheld.")
                continue
            if q.data_timestamp < pd.Timestamp(pos["valuation_timestamp"]).to_pydatetime():
                continue
            # Only completed bars before this quote contribute to the trailing
            # high and ATR; replay must not see the remainder of its data bundle.
            end = pd.Timestamp(q.data_timestamp.date()) - pd.Timedelta(days=1)
            high = self.bundle.high.loc[pos["entry"]:end, ticker].dropna() if ticker in self.bundle.tickers else pd.Series(dtype=float)
            peak = max(float(pos["entry_px"]), q.price, self._high_water.get(ticker, 0),
                       float(high.max()) if len(high) else 0)
            self._high_water[ticker] = peak
            stop = pos.get("stop")
            if stop is not None and q.price <= stop:
                self._alert("stop:" + ticker, "Held position is below its planned stop. Alert only; no order placed.", ticker=ticker)
            else:
                self.alerts.pop("stop:" + ticker, None)
            if ticker not in self.bundle.tickers:
                continue
            h = self.bundle.high.loc[:end, ticker].tail(int(self.cfg.execution.atr_period) + 1)
            low = self.bundle.low.loc[h.index, ticker]
            prev = self.bundle.close.loc[:end, ticker].shift(1).reindex(h.index)
            atr = pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1).tail(int(self.cfg.execution.atr_period)).mean()
            trailing = peak - float(self.cfg.execution.trailing_atr_mult) * atr
            pos["trailing_stop"] = trailing
            if math.isfinite(trailing) and q.price <= trailing:
                self._alert("trailing:" + ticker, "Held position breached its ATR trailing stop. Alert only; no order placed.", ticker=ticker)
            else:
                self.alerts.pop("trailing:" + ticker, None)

    async def _run(self):
        try:
            await asyncio.to_thread(self._bootstrap)
            if self.feed is None:
                kwargs = {"bundle": self.bundle,
                          "replay_start": pd.Timestamp(self.meta["asof"]) + pd.Timedelta(days=1)} if self.feed_name == "synthetic" else {}
                self.feed = make_feed(self.feed_name, **kwargs)
            self.status = "ready"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never expose provider exception strings (may contain keys/URLs).
            self.status, self.error = "error", str(exc) if isinstance(exc, SetupError) else "Initialization failed; check data/configuration and local logs."
            log.error("Market initialization failed (%s)", type(exc).__name__)
            return
        last_scan = time.monotonic()
        self._next_scan_at = (utc_now() + timedelta(seconds=max(60, float(self.cfg.live.get("scan_seconds", 3600))))).isoformat()
        failures = 0
        while True:
            try:
                tickers = sorted({r["ticker"] for r in self.rows} | self._position_tickers)
                quotes = await asyncio.wait_for(self.feed.poll(tickers), timeout=30)
                self.quotes.update({q.ticker: q for q in quotes})
                self._last_poll, self.feed_error = utc_now().isoformat(), None
                self.alerts.pop("feed", None)
                failures = 0
                self._portfolio = await asyncio.to_thread(self._read_portfolio)
                await asyncio.to_thread(self._check_risk)
                if time.monotonic() - last_scan >= max(60, float(self.cfg.live.get("scan_seconds", 3600))):
                    if self.feed_name != "synthetic" and time.monotonic() - self._history_loaded >= float(self.cfg.live.get("history_refresh_seconds", 86400)):
                        self.bundle = await asyncio.to_thread(load_bundle, self.cfg, use_cache=False)
                        self._history_loaded = time.monotonic()
                    await asyncio.to_thread(self._scan)
                    last_scan = time.monotonic()
                    self._next_scan_at = (utc_now() + timedelta(seconds=max(60, float(self.cfg.live.get("scan_seconds", 3600))))).isoformat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                self.feed_error = f"Update failed ({type(exc).__name__}); retaining last timestamps while retrying."
                self._alert("feed", self.feed_error, "error")
                log.warning("Market update failed (%s)", type(exc).__name__)
            await asyncio.sleep(min(120, max(1, float(self.cfg.live.get("poll_seconds", 5))) * 2 ** min(failures, 4)))

    def snapshot(self):
        now = utc_now()
        expected = {r["ticker"] for r in self.rows}
        effective_feed_error = self.feed_error or getattr(self.feed, "last_error", None)
        connected = getattr(self.feed, "connected", True)
        quotes = [replace(q, connected=False) if effective_feed_error or not connected else q
                  for q in self.quotes.values()]
        fresh = global_freshness(quotes, now)
        missing = expected - self.quotes.keys()
        if missing or effective_feed_error or not connected:
            fresh = {"state": "red", "label": "Incomplete or disconnected quote feed", "icon": "!", "live": False}
        rows = copy.deepcopy(self.rows)
        quote_dicts = {q.ticker: q.to_dict(now) for q in quotes}
        for row in rows:
            quote = quote_dicts.get(row["ticker"])
            signal_time = datetime.fromisoformat(row.get("signal_timestamp", row["signal_asof"]))
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            row.update(price=quote["price"] if quote else None, quote=quote,
                       signal_age_seconds=max(0, (now - signal_time).total_seconds()))
        validated = bool(self.bundle is not None and system_validated(self.spec, self.cfg, self.bundle.meta))
        flags = []
        if not validated:
            flags.append("System not validated: confidence capped at Low. No real-market advantage established.")
        if self.spec.get("provisional"):
            flags.append("Research metrics unavailable for the provisional baseline; run the research workflow before interpreting model quality.")
        mkt = market_status(now)
        decay = []
        summary_path = Path(self.cfg.paths.artifacts) / "validation_summary.json"
        summary = {}
        if summary_path.exists():
            try:
                candidate = json.loads(summary_path.read_text())
                if candidate.get("config_hash") == self.cfg.hash():
                    summary = candidate
                    flags.extend(summary.get("red_flags", []))
                    decay = summary.get("signal_decay", [])
            except (OSError, ValueError):
                flags.append("Saved validation summary is unreadable")
        return clean({"status": self.status, "error": self.error, "mode": self.feed_name,
            "updated_at": self._last_poll, "scan": {**self.meta, "scored_at": self.scored_at,
                "universe_count": len(rows), "next_scan_at": self._next_scan_at,
                "feature_updates": {"monthly_features": {"updated_at": rows[0].get("signal_timestamp") if rows else None,
                                        "label": "Monthly model inputs", "description": "Available at the signal close. Refitted for each new completed month."},
                                    "quote_risk_checks": {"updated_at": self._last_poll,
                                        "label": "Prices and risk", "description": "Prices refresh independently. Real-feed corporate-action comparisons remain withheld."},
                                    "news": {"label": "News strategy disabled", "description": "No validated live news source connected; net-cost validation failed."}}},
            "freshness": fresh, "market": {**mkt, "status": mkt["state"]},
            "quotes": list(quote_dicts.values()), "recommendations": rows,
            "portfolio": self._portfolio, "feed": {"provider": self.feed_name,
                "status": "error" if effective_feed_error else "connected" if quotes and connected else "waiting",
                "poll_interval_seconds": max(1, float(self.cfg.live.get("poll_seconds", 5))),
                "error": effective_feed_error, "coverage": getattr(self.feed, "coverage", "Configured eligible universe"),
                "expected_symbols": len(expected), "received_symbols": len(expected & self.quotes.keys()),
                "missing_symbols": sorted(missing)},
            "validation": {"system_validated": validated,
                "data_source": self.bundle.meta.get("source") if self.bundle is not None else self.cfg.data.provider,
                "dsr": self.spec.get("dsr"), "pbo": self.spec.get("pbo"), "n_trials": summary.get("n_trials"),
                "holdout_status": "used_failed" if self.feed_name == "synthetic" else "passed" if validated else "not_validated",
                "red_flags": flags, "signal_decay": decay, "metrics": summary.get("metrics", {})},
            "alerts": list(self.alerts.values())[-100:], "disclaimer": DISCLAIMER})

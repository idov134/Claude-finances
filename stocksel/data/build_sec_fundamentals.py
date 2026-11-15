"""Point-in-time fundamentals and earnings from SEC EDGAR XBRL ("company facts"), free.

    python -m stocksel.data.build_sec_fundamentals --data data_csv

Writes into the canonical CSV folder (see providers/csv_provider.py):
  fundamentals.csv  ticker, period_end, publish_date, total_assets, book_equity, gross_profit, net_income,
                    revenue, eps        (quarterly flows; balance-sheet values at quarter end)
  earnings.csv      ticker, announce_ts, eps_actual, eps_consensus (empty -> seasonal-random-walk SUE), period_end
  pit_fundamentals.flag, sec_info.json

Point-in-time rules
* every value is the **first-filed** figure for its period (later restatements are ignored), and is
  available from its SEC **filing date** (`filed`), never from the period end;
* quarterly flows come from 3-month facts; missing quarters are derived from year-to-date differences
  sharing the fiscal-year start (Q4 = FY - 9M), dated by the later filing;
* a row's publish_date is the latest filing date among its fields (conservative);
* earnings are usable after the filing date (filed 23:59 New York), later than the press release, so
  PEAD measured this way is weaker than with announcement timestamps, but never early;
* a company that moved to a new SEC registrant keeps its earlier history only if the successor's
  comparative figures verify the predecessor (merge_predecessor); the feature layer additionally lets
  only the newest fiscal period known at each date enter the join (features/fundamental.py).

Coverage: XBRL starts 2009 (large filers) - 2011 (all). US-GAAP filers only. Only current tickers can be
mapped to a CIK, which matches the survivor-only price data. The SEC asks automated clients to identify
themselves; this uses a generic client name (set SEC_USER_AGENT to use your own contact).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = os.environ.get("SEC_USER_AGENT", "stocksel-research research@example.com")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

FLOWS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
}
STOCKS = {
    "total_assets": ["Assets"],
    "book_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}

# Companies that moved to a new SEC registrant (holding-company reorganization, redomicile), newest
# predecessor first. The current ticker map only knows the new registrant, which loses the history.
# These are CANDIDATES: a predecessor is merged only if verify_predecessor() finds that the successor's
# comparative figures match it (see there); rejected candidates are listed in sec_info.json.
PREDECESSORS = {"XOM": [34088], "BLK": [1364742], "DIS": [1001039], "GOOGL": [1288776], "GOOG": [1288776],
                "MDT": [64670], "AVGO": [1649338, 1441634], "CI": [701221], "APA": [6769], "XRX": [108772],
                "DD": [30554], "FTI": [1135152], "ETN": [31277], "ICE": [1174746], "PRGO": [820096],
                "BG": [1144519], "DOW": [29915], "NWSA": [1308161]}


def _get(url: str, session: requests.Session, tries: int = 5):
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                time.sleep(2 ** k)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** k)
    return None


def _facts(gaap: dict, tags: list[str], unit_prefix: str = "USD") -> pd.DataFrame:
    rows = []
    for priority, tag in enumerate(tags):
        for unit, facts in (gaap.get(tag, {}).get("units", {}) or {}).items():
            if not unit.startswith(unit_prefix):
                continue
            for f in facts:
                if f.get("form", "").startswith(("10-K", "10-Q")) and "filed" in f:
                    rows.append({"start": f.get("start"), "end": f["end"], "val": f["val"], "filed": f["filed"],
                                 "priority": priority})
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed"])
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # first-filed figure per period (as originally reported); ties -> preferred tag
    df = df.sort_values(["filed", "priority"]).drop_duplicates(["start", "end"] if df["start"].notna().any()
                                                               else ["end"], keep="first")
    return df


def quarterly_flows(df: pd.DataFrame) -> pd.Series:
    """{quarter_end: (value, filed)} from 3-month facts plus year-to-date differences."""
    if df.empty:
        return pd.Series(dtype=object)
    df = df.dropna(subset=["start", "end"]).copy()
    df["days"] = (df["end"] - df["start"]).dt.days
    out = {}
    for r in df[(df["days"] >= 80) & (df["days"] <= 100)].itertuples():
        out[r.end] = (float(r.val), r.filed)
    for _, g in df[(df["days"] >= 80) & (df["days"] <= 380)].groupby("start"):
        g = g.sort_values("end")
        prev = None
        for r in g.itertuples():
            if prev is not None and 80 <= (r.end - prev.end).days <= 100 and r.end not in out:
                out[r.end] = (float(r.val) - float(prev.val), max(r.filed, prev.filed))
            prev = r
    return pd.Series(out).sort_index()


def instants(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    df = df.sort_values("filed").drop_duplicates("end", keep="first")
    return pd.Series({r.end: (float(r.val), r.filed) for r in df.itertuples()}).sort_index()


def merge_predecessor(gaap_new: dict, gaap_old: dict) -> tuple[dict, dict]:
    """Prepend a predecessor registrant's history, if the two are verifiably the same business.

    Cutover = the successor's first filing. The successor's first reports restate earlier quarters as
    comparatives; the predecessor is accepted only if at least 2 such quarters exist and at least 75%
    of them match its own net income within 1%. Only predecessor facts for periods ending before the
    cutover are added, and the first-filed rule then keeps the predecessor's original numbers.
    """
    ni_new = _facts(gaap_new, FLOWS["net_income"])
    if ni_new.empty:
        return gaap_new, {"verified": False, "why": "successor has no net income"}
    cut = ni_new["filed"].min()
    qn, qo = quarterly_flows(ni_new), quarterly_flows(_facts(gaap_old, FLOWS["net_income"]))
    common = [d for d in qn.index.intersection(qo.index) if d < cut]
    hits = [abs(qn[d][0] - qo[d][0]) <= 0.01 * max(abs(qo[d][0]), 1.0) for d in common]
    ok = len(common) >= 2 and np.mean(hits) >= 0.75
    ev = {"verified": bool(ok), "cutover": str(cut.date()), "overlap_quarters": len(common),
          "matching": int(sum(hits))}
    if not ok:
        return gaap_new, ev
    cut_s = cut.strftime("%Y-%m-%d")
    merged = {tag: {"units": {u: list(v) for u, v in d.get("units", {}).items()}} for tag, d in gaap_new.items()}
    for tag, d in gaap_old.items():
        for unit, facts in (d.get("units", {}) or {}).items():
            old = [f for f in facts if f.get("end", "9999") < cut_s]
            merged.setdefault(tag, {"units": {}})["units"].setdefault(unit, []).extend(old)
    return merged, ev


def company_rows(ticker: str, facts: dict) -> list[dict]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    if not gaap:
        return []
    flows = {k: quarterly_flows(_facts(gaap, tags, "USD/shares" if k == "eps" else "USD")) for k, tags in FLOWS.items()}
    stocks = {k: instants(_facts(gaap, tags)) for k, tags in STOCKS.items()}
    ends = sorted(set().union(*[s.index for s in flows.values()]) | set().union(*[s.index for s in stocks.values()]))
    rows = []
    for e in ends:
        rec, filed = {"ticker": ticker, "period_end": e}, []
        for k, s in {**flows, **stocks}.items():
            if e in s.index:
                rec[k], f = s[e]
                filed.append(f)
            else:
                rec[k] = np.nan
        if pd.isna(rec.get("gross_profit")) and pd.notna(rec.get("revenue")) and pd.notna(rec.get("cost_of_revenue")):
            rec["gross_profit"] = rec["revenue"] - rec["cost_of_revenue"]
        rec.pop("cost_of_revenue", None)
        if pd.isna(rec.get("net_income")) and pd.isna(rec.get("eps")):
            continue
        rec["publish_date"] = max(filed)
        rows.append(rec)
    return rows


def write_earnings(f: pd.DataFrame, data_dir: Path) -> None:
    """earnings.csv from the fundamentals rows: usable after the filing date (23:59 New York)."""
    e = f.dropna(subset=["eps"])
    ts = (pd.to_datetime(e["publish_date"]) + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC")
    pd.DataFrame({"ticker": e["ticker"], "announce_ts": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "eps_actual": e["eps"], "eps_consensus": np.nan,
                  "period_end": pd.to_datetime(e["period_end"]).dt.strftime("%Y-%m-%d")}).to_csv(
        data_dir / "earnings.csv", index=False)


def build(data_dir: Path, sleep: float = 0.15, limit: int | None = None) -> dict:
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    tickers = sorted(pd.read_csv(data_dir / "sectors.csv")["ticker"].unique())
    if limit:
        tickers = tickers[:limit]
    m = _get(TICKERS_URL, session) or {}
    cik = {}
    for v in m.values():
        cik[v["ticker"].upper()] = int(v["cik_str"])
    rows, mapped, with_facts, pred = [], 0, 0, {}
    for i, t in enumerate(tickers):
        c = cik.get(t) or cik.get(t.replace("-", ".")) or cik.get(t.replace("-", ""))
        if c is None:
            continue
        mapped += 1
        facts = _get(FACTS_URL.format(cik=c), session)
        time.sleep(sleep)
        if not facts:
            continue
        for p in PREDECESSORS.get(t, []):
            old = _get(FACTS_URL.format(cik=p), session)
            time.sleep(sleep)
            if not old:
                pred[f"{t}<-{p}"] = {"verified": False, "why": "not found"}
                continue
            g, ev = merge_predecessor(facts.get("facts", {}).get("us-gaap", {}),
                                      old.get("facts", {}).get("us-gaap", {}))
            pred[f"{t}<-{p}"] = {**ev, "name": old.get("entityName")}
            facts = {**facts, "facts": {**facts.get("facts", {}), "us-gaap": g}}
        r = company_rows(t, facts)
        if r:
            with_facts += 1
            rows.extend(r)
        if (i + 1) % 100 == 0:
            print(f"[sec] {i + 1}/{len(tickers)} mapped={mapped} with_facts={with_facts}", flush=True)
    f = pd.DataFrame(rows)
    f = f[["ticker", "period_end", "publish_date", "total_assets", "book_equity", "gross_profit", "net_income",
           "revenue", "eps"]].sort_values(["ticker", "period_end"])
    f.to_csv(data_dir / "fundamentals.csv", index=False, date_format="%Y-%m-%d")
    write_earnings(f, data_dir)
    (data_dir / "pit_fundamentals.flag").write_text("SEC XBRL first-filed values, available from filing date\n")
    lag = (pd.to_datetime(f["publish_date"]) - pd.to_datetime(f["period_end"])).dt.days
    info = {"built": time.strftime("%Y-%m-%d %H:%M"), "tickers": len(tickers), "mapped_to_cik": mapped,
            "with_us_gaap_facts": with_facts, "rows": int(len(f)),
            "first_publish": str(pd.to_datetime(f["publish_date"]).min().date()),
            "median_filing_lag_days": float(lag.median()), "predecessors": pred,
            "share_rows_with": {c: float(f[c].notna().mean()) for c in
                                ("total_assets", "book_equity", "gross_profit", "net_income", "revenue", "eps")}}
    (data_dir / "sec_info.json").write_text(json.dumps(info, indent=2))
    print(f"[sec] done: {json.dumps(info)}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stocksel.data.build_sec_fundamentals")
    ap.add_argument("--data", default="data_csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rewrite-earnings", action="store_true", help="only rebuild earnings.csv from fundamentals.csv")
    args = ap.parse_args(argv)
    if args.rewrite_earnings:
        write_earnings(pd.read_csv(Path(args.data) / "fundamentals.csv"), Path(args.data))
        return
    build(Path(args.data), limit=args.limit)


if __name__ == "__main__":
    main()

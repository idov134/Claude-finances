"""Run the technical-rules pipeline on random-walk markets (no signal) to calibrate false positives.

    python -m stocksel.experiments.technical_null_calibration OUT.json 101,202,303,404,505

Fat-tailed (t4) random walks with drift for 11 "ETFs" and 120 stocks, a random VIX path, the same
dates and the same pipeline as the real study. Results: docs/technical_rules/null_calibration/.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import stocksel.experiments.technical_rules as tr
tr.BOOT["n"] = 50
out = {}
for seed in [int(x) for x in sys.argv[2].split(',')]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1999-01-04", "2026-10-30")
    n = len(idx)
    def walk(cols, vol):
        r = rng.standard_t(4, size=(n, len(cols))) * vol / np.sqrt(2) + 0.0003
        c = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
        o = c.shift(1) * np.exp(rng.normal(0, 0.002, size=c.shape)); o.iloc[0] = c.iloc[0]
        return o, c
    eo, ec = walk(tr.ETFS, 0.013)
    tick = [f"T{i:03d}" for i in range(120)]
    so, sc = walk(tick, 0.02)
    mcap = sc * rng.uniform(1e9, 5e10, size=len(tick))
    member = pd.DataFrame(True, index=idx, columns=tick)
    sectors = pd.Series(["Technology"] * 40 + ["Energy"] * 30 + ["Healthcare"] * 50, index=tick)
    vix = pd.Series(np.clip(20 + np.cumsum(rng.normal(0, 1, n)) * 0.3, 10, 60), index=idx)
    inp = tr.Inputs(etf_open=eo, etf_close=ec, stock_open=so, stock_close=sc, stock_mcap=mcap, member=member,
                    sectors=sectors, rf=pd.Series(0.02, index=idx), vix=vix,
                    prev_alpha_sharpes=np.array([]), n_previous=57)
    with tempfile.TemporaryDirectory() as d:
        res = tr.run(inp, Path(d), n_shifts={"etf": 200, "stock": 30})
        T = pd.read_csv(Path(d) / "trials.csv")
        T.to_csv(Path(sys.argv[1]).with_name(f"null_trials_{seed}.csv"), index=False)
    out[seed] = {"global": res["global"], "edge_confirmed": res["edge_confirmed"],
                 "risk_control": res["risk_control"],
                 "promising_ic": [k for k, v in res["cross_sectional_ic"].items() if v["promising"]],
                 "max_dsr": float(T["full_dsr_alpha"].max())}
    print(seed, json.dumps(out[seed], default=str), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))

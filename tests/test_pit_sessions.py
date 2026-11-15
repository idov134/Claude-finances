"""Exchange close times are also information-availability boundaries."""
import numpy as np
import pandas as pd

from stocksel.data.pit import asof_wide, availability, decision_timestamps
from stocksel.features.news import effective_day_positions


def test_early_close_earnings_and_news_wait_for_next_session():
    dates = pd.DatetimeIndex(["2023-11-24", "2023-11-27"])
    stamps = pd.Series(pd.to_datetime(["2023-11-24T17:59:00Z", "2023-11-24T18:00:00Z",
                                      "2023-11-24T19:00:00Z"]))
    # Nov 24 closed at 13:00 EST; equality is not before the decision.
    assert effective_day_positions(stamps, dates, "America/New_York", "16:00").tolist() == [0, 1, 1]
    events = pd.DataFrame({"ticker": ["A"], "announce_ts": [stamps.iloc[2]], "x": [99.]})
    before = asof_wide(events, "announce_ts", ["x"], dates, ["A"])["x"]
    assert np.isnan(before.iloc[0, 0]) and before.iloc[1, 0] == 99
    events["x"] *= 100
    after = asof_wide(events, "announce_ts", ["x"], dates, ["A"])["x"]
    assert np.isnan(after.iloc[0, 0])  # changing post-close news cannot change the prior feature


def test_naive_intraday_availability_is_local_and_not_normalized():
    dates = pd.DatetimeIndex(["2023-11-27", "2023-11-28"])
    events = pd.DataFrame({"ticker": ["A"], "announce_ts": [pd.Timestamp("2023-11-27 16:30")], "x": [1.]})
    result = asof_wide(events, "announce_ts", ["x"], dates, ["A"])["x"]
    assert np.isnan(result.iloc[0, 0]) and result.iloc[1, 0] == 1


def test_aware_publications_obey_configured_lag():
    ts = pd.Series([pd.Timestamp("2023-11-27 12:00", tz="America/New_York")])
    assert availability(ts, "America/New_York", 1).iloc[0] == pd.Timestamp("2023-11-28 12:00")


def test_regular_close_and_custom_market_clock_remain_explicit():
    dates = pd.DatetimeIndex(["2023-11-24", "2023-11-27"])
    assert decision_timestamps(dates).hour.tolist() == [13, 16]
    assert decision_timestamps(dates, "15:30", "Asia/Jerusalem").hour.tolist() == [15, 15]
    assert decision_timestamps(dates, "15:30", "Asia/Jerusalem").minute.tolist() == [30, 30]


def test_dst_news_alignment_uses_exchange_local_time():
    dates = pd.DatetimeIndex(["2023-11-03", "2023-11-06", "2023-11-07"])
    # Before DST ends, close is 20:00 UTC; afterward it is 21:00 UTC.
    stamps = pd.Series(pd.to_datetime(["2023-11-03T20:00:00Z", "2023-11-06T20:59:00Z",
                                      "2023-11-06T21:00:00Z"]))
    assert effective_day_positions(stamps, dates, "America/New_York", "16:00").tolist() == [1, 1, 2]


def test_ttm_waits_for_each_constituent_publication():
    from stocksel.features.fundamental import _ttm
    f = pd.DataFrame({
        "ticker": ["A"] * 4,
        "period_end": pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]),
        # Q3 was delayed until after Q4. It cannot enter Q4 TTM in February.
        "publish_date": pd.to_datetime(["2022-05-01", "2022-08-01", "2023-03-01", "2023-02-01"]),
        "gross_profit": [10., 20., 30., 40.], "net_income": [10., 20., 30., 40.],
        "revenue": [10., 20., 30., 40.],
    })
    dates = pd.DatetimeIndex(["2023-02-28", "2023-03-02"])
    aggregate = _ttm(f.copy())
    first = asof_wide(aggregate, "publish_date", ["net_income_ttm"], dates, ["A"], lag_days=1)["net_income_ttm"]
    f.loc[2, "net_income"] = 3000.
    second = asof_wide(_ttm(f.copy()), "publish_date", ["net_income_ttm"], dates, ["A"], lag_days=1)["net_income_ttm"]
    assert aggregate.iloc[-1].publish_date == pd.Timestamp("2023-03-01")
    assert np.isnan(first.iloc[0, 0]) and np.isnan(second.iloc[0, 0])
    assert first.iloc[1, 0] == 100.
    assert second.iloc[1, 0] == 3070.


def test_ttm_requires_four_consecutive_quarters():
    from stocksel.features.fundamental import _ttm
    f = pd.DataFrame({
        "ticker": ["A"] * 5,
        # 2022-09-30 is missing: the four rows ending 2023-03-31 span five quarters
        "period_end": pd.to_datetime(["2022-03-31", "2022-06-30", "2022-12-31", "2023-03-31", "2023-06-30"]),
        "publish_date": pd.to_datetime(["2022-05-01", "2022-08-01", "2023-02-01", "2023-05-01", "2023-08-01"]),
        "gross_profit": [1., 2., 3., 4., 5.], "net_income": [1., 2., 3., 4., 5.], "revenue": [1., 2., 3., 4., 5.],
    })
    out = _ttm(f.copy())
    assert out["net_income_ttm"].isna().all()
    f2 = f.copy()
    f2.loc[2, "period_end"] = pd.Timestamp("2022-09-30")
    f2.loc[3, "period_end"] = pd.Timestamp("2022-12-31")
    f2.loc[4, "period_end"] = pd.Timestamp("2023-03-31")
    out2 = _ttm(f2.copy())
    assert out2["net_income_ttm"].tolist()[3:] == [10., 14.]


def test_seasonal_sue_skips_gaps_when_period_end_known():
    from stocksel.features.fundamental import seasonal_surprise
    ends = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30",
            "2021-12-31", "2022-06-30", "2022-09-30"]          # 2022-03-31 missing
    e = pd.DataFrame({"ticker": ["A"] * len(ends), "period_end": ends,
                      "eps_actual": [1., 1.2, 0.9, 1.1, 1.3, 1.0, 1.4, 1.2, 1.5, 1.6]})
    gated = seasonal_surprise(e)
    plain = seasonal_surprise(e.drop(columns="period_end"))
    # with the gap, four rows before 2022-06-30 is 2021-03-31, not the same quarter -> dropped
    assert np.isnan(gated.iloc[8]) and np.isnan(gated.iloc[9])
    assert np.isfinite(plain.iloc[8])                      # without period_end the wrong quarter is used
    assert np.allclose(gated.iloc[:8].dropna(), plain.iloc[:8].dropna())


def test_late_comparative_does_not_overwrite_newer_period():
    from stocksel.features.fundamental import newest_period_only
    ev = pd.DataFrame({"ticker": ["A", "A", "A", "A"],
                       "period_end": pd.to_datetime(["2011-03-31", "2011-06-30", "2010-06-30", "2011-09-30"]),
                       # the 2010-06-30 comparative is first reported in the 2011 Q3 filing season
                       "publish_date": pd.to_datetime(["2011-05-01", "2011-08-01", "2011-08-15", "2011-11-01"]),
                       "net_income": [1., 2., 99., 3.]})
    out = newest_period_only(ev, "publish_date")
    assert out["net_income"].tolist() == [1., 2., 3.]
    dates = pd.DatetimeIndex(["2011-08-31"])
    j = asof_wide(out, "publish_date", ["net_income"], dates, ["A"], lag_days=1)["net_income"]
    assert j.iloc[0, 0] == 2.0
    # same filing carrying an older comparative: the newer period wins the tie
    tie = pd.DataFrame({"ticker": ["A", "A"], "period_end": pd.to_datetime(["2011-06-30", "2010-06-30"]),
                        "publish_date": pd.to_datetime(["2011-08-01", "2011-08-01"]), "net_income": [2., 99.]})
    j2 = asof_wide(newest_period_only(tie, "publish_date"), "publish_date", ["net_income"], dates, ["A"],
                   lag_days=1)["net_income"]
    assert j2.iloc[0, 0] == 2.0


def _gaap(ni: dict[str, tuple[str, str, float, str]]) -> dict:
    facts = [{"start": a, "end": b, "val": v, "filed": f, "form": "10-Q"} for a, b, v, f in ni.values()]
    return {"NetIncomeLoss": {"units": {"USD": facts}}}


def test_predecessor_merged_only_when_comparatives_match():
    from stocksel.data.build_sec_fundamentals import company_rows, merge_predecessor
    old = _gaap({1: ("2020-01-01", "2020-03-31", 10., "2020-05-01"), 2: ("2020-04-01", "2020-06-30", 11., "2020-08-01"),
                 3: ("2020-07-01", "2020-09-30", 12., "2020-11-01"),
                 4: ("2021-01-01", "2021-03-31", 999., "2021-05-01")})   # subsidiary keeps filing after cutover
    new = _gaap({1: ("2020-04-01", "2020-06-30", 11., "2021-02-01"), 2: ("2020-07-01", "2020-09-30", 12., "2021-02-01"),
                 3: ("2021-01-01", "2021-03-31", 13., "2021-05-01")})
    merged, ev = merge_predecessor(new, old)
    assert ev["verified"] and ev["overlap_quarters"] == 2
    rows = {r["period_end"].strftime("%Y-%m-%d"): r for r in company_rows("X", {"facts": {"us-gaap": merged}})}
    assert rows["2020-03-31"]["net_income"] == 10. and rows["2020-06-30"]["publish_date"] == pd.Timestamp("2020-08-01")
    assert rows["2021-03-31"]["net_income"] == 13.          # predecessor facts after the cutover are ignored
    other = _gaap({1: ("2020-04-01", "2020-06-30", 50., "2020-08-01"), 2: ("2020-07-01", "2020-09-30", 60., "2020-11-01")})
    same, ev2 = merge_predecessor(new, other)
    assert not ev2["verified"] and same is new

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

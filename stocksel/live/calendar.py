"""Approximate NYSE trading calendar for live order dates (backtests use the data's own calendar)."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
                                    nearest_workday)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_BDAY = pd.offsets.CustomBusinessDay(calendar=NYSEHolidayCalendar())


def next_trading_day(d) -> pd.Timestamp:
    return (pd.Timestamp(d).normalize() + NYSE_BDAY)


def first_trading_day_after_months(d, months: int) -> pd.Timestamp:
    """First trading day after the last trading day of the month ``months`` ahead (next rebalance open)."""
    end = pd.Timestamp(d) + pd.offsets.MonthEnd(months)
    last = end if NYSE_BDAY.is_on_offset(end) else end - NYSE_BDAY
    return last + NYSE_BDAY

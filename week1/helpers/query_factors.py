import numpy as np
import pandas as pd
import tushare as ts
import yaml
import sys
import os

from pandas.tseries.offsets import BDay


PATH = os.path.abspath(__file__)
for _ in range(4):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

from FactorAnalysis.week1.helpers.utils import load_pro

PRO = load_pro()


def query_daily_basic(ts_code: str, trade_date: str, fields=None, **kwargs):
    """
    Default: total_mv, pe, pb, turnover_rate
    """
    if fields is None:
        fields = "total_mv, pe, pb, turnover_rate"
    data_df = PRO.daily_basic(ts_code=ts_code, trade_date=trade_date, fields=fields)

    return data_df


def query_daily(ts_code: str, trade_date: str, **kwargs):
    """
    Default: reversal_rate, volatility
    kwargs: reversal_duration: int, vol_duration: int

    reversal_rate is the pct_chg over a period of time
    """
    def compute_start_date(duration: int):
        trade_date_dt = pd.to_datetime(trade_date)
        start_date = trade_date_dt - BDay(duration)

        return start_date.strftime("%Y%m%d")

    fields = ["reversal_rate", "volatility"]
    df_out = pd.DataFrame(columns=fields)

    reversal_duration = kwargs.get("reversal_duration", 5)
    vol_duration = kwargs.get("vol_duration", 30)

    # reversal_rate
    field = "reversal_rate"
    start_date = compute_start_date(reversal_duration)
    # print(start_date)
    end_date = trade_date
    data_df = PRO.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    # trade_date: recent first
    df_out.loc[0, field] = (data_df.close.iloc[0] - data_df.close.iloc[-1]) / data_df.close.iloc[-1] * 100

    # volatility
    field = "volatility"
    start_date = compute_start_date(vol_duration)
    end_date = trade_date
    data_df = PRO.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_out.loc[0, field] = data_df.pct_chg.std()

    return df_out


def query_fina_indicator(ts_code: str, trade_date: str, fields=None, **kwargs):
    """
    Default: roe, netprofit_yoy, or_yoy, assets_yoy, equity_yoy
    """
    if fields is None:
        fields = "roe, netprofit_yoy, or_yoy, assets_yoy, equity_yoy"
    data_df = PRO.fina_indicator(ts_code=ts_code)

    return data_df.iloc[0:1][fields.split(", ")]

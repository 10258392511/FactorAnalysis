import numpy as np
import pandas as pd
import sqlite3
import sys
import os

from pandas.tseries.offsets import BDay
from datetime import datetime as dt
from tqdm import tqdm


PATH = os.path.abspath(__file__)
for _ in range(4):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

from FactorAnalysis.week1.helpers.utils import load_pro
from FactorAnalysis.week1.helpers.data_processing import read_from_db

PRO = load_pro()


def query_daily_basic(ts_code: str, trade_date: str, fields=None, **kwargs):
    """
    Default: total_mv, pe, pb, turnover_rate
    """
    if fields is None:
        fields = "total_mv, pe, pb, turnover_rate"
    data_df = PRO.daily_basic(ts_code=ts_code, trade_date=trade_date, fields=fields).fillna(value=np.nan)

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
    data_df = PRO.daily(ts_code=ts_code, start_date=start_date, end_date=end_date).fillna(value=np.nan)
    # trade_date: recent first
    df_out.loc[0, field] = (data_df.close.iloc[0] - data_df.close.iloc[-1]) / data_df.close.iloc[-1] * 100

    # volatility
    field = "volatility"
    start_date = compute_start_date(vol_duration)
    end_date = trade_date
    data_df = PRO.daily(ts_code=ts_code, start_date=start_date, end_date=end_date).fillna(value=np.nan)
    df_out.loc[0, field] = data_df.pct_chg.std()

    return df_out


def query_fina_indicator(ts_code: str, trade_date: str, fields=None, **kwargs):
    """
    Default: roe, netprofit_yoy, or_yoy, assets_yoy, equity_yoy
    """
    if fields is None:
        fields = "roe, netprofit_yoy, or_yoy, assets_yoy, equity_yoy"
    data_df = PRO.fina_indicator(ts_code=ts_code).fillna(value=np.nan)

    return data_df.iloc[0:1][fields.split(", ")]


def query_financial_statements(ts_code: str, trade_date: str, **kwargs):
    """
    Default: gross_profit_margin, net_profit_margin, operating_cash_flow_to_net_income,
    operating_cash_flow_to_revenue, current_ratio, cash_current_liability_ratio, cash_liability_ratio,
    long_term_liability_operating_cash_flow_ratio
    """
    income_fields = "revenue, operate_profit, n_income"
    bs_fields = "total_cur_assets, total_cur_liab, total_liab, total_assets, lt_borr"
    cashflow_fields = "n_cashflow_act, net_profit"

    df_income = PRO.income(ts_code=ts_code, fields=income_fields).iloc[0].fillna(value=np.nan)
    df_bs = PRO.balancesheet(ts_code=ts_code, fields=bs_fields).iloc[0].fillna(value=np.nan)
    df_cf = PRO.cashflow(ts_code=ts_code, fields=cashflow_fields).iloc[0].fillna(value=np.nan)

    data_out = {
        "gross_profit_margin": df_income["operate_profit"] / df_income["revenue"],
        "net_profit_margin": df_income["n_income"] / df_income["revenue"],
        "operating_cash_flow_to_net_income": df_cf["n_cashflow_act"] / df_income["n_income"],
        "operating_cash_flow_to_revenue": df_cf["n_cashflow_act"] / df_income["revenue"],
        "current_ratio": df_bs["total_cur_assets"] / df_bs["total_cur_liab"],
        "cash_current_liability_ratio": df_cf["n_cashflow_act"] / df_bs["total_cur_liab"],
        "cash_liability_ratio": df_cf["n_cashflow_act"] / df_bs["total_liab"],
        "long_term_liability_operating_cash_flow_ratio": df_bs["lt_borr"] / df_cf["n_cashflow_act"]
    }

    return pd.DataFrame(data_out, index=[0])


def query_factors(ts_code: str, trade_date: str, **kwargs):
    """
    kwargs: See query_daily(.)
    """
    funcs = [query_daily_basic, query_daily, query_fina_indicator, query_financial_statements]
    df_all = []
    for func_iter in funcs:
        df_all.append(func_iter(ts_code, trade_date, **kwargs))

    df_out = pd.concat(df_all, axis=1)

    return df_out


def save_factors_A_share(db_filename: str, date: str, log_dir: str):
    """
    date: "YYYYMMDD"
    date, ts_code | status -> date, ts_code | status, factors...
    """
    timestamp = dt.today().strftime("%Y%m%d_%H%M%S")
    log_file = open(os.path.join(log_dir, f"{timestamp}.log"), "a")
    df = read_from_db("A_share", db_filename, parse_dates=False)
    df_on_date = df.loc[(date,)]
    for ts_code in tqdm(df_on_date.index):
        try:
            df_factors = query_factors(ts_code, date)
            for col in df_factors.columns:
                df.loc[(date, ts_code), col] = df_factors.iloc[0][col]
        except Exception as err:
            print(f"Unsuccessful with error {err}: {ts_code}", file=log_file)

    conn = sqlite3.connect(db_filename)
    df.to_sql("A_share", conn, if_exists="replace")

    conn.close()
    log_file.close()

    return df


def save_factors_others(db_filename: str, table_name: str):
    """
    JOIN with updated A_share
    """
    df = read_from_db(table_name, db_filename, parse_dates=False)
    df = df[["status"]]
    all_stocks = read_from_db("A_share", db_filename, parse_dates=False)
    df_out = df.merge(all_stocks.drop(columns="status"), left_index=True, right_index=True, how="left")

    conn = sqlite3.connect(db_filename)
    df_out.to_sql(table_name, conn, if_exists="replace")

    conn.close()

    return df_out

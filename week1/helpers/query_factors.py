import numpy as np
import pandas as pd
import sqlite3
import sys
import os

from pandas.tseries.offsets import BDay
from datetime import datetime as dt
from typing import Iterable
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


def get_all_stocks():
    all_stocks = PRO.stock_basic(ts_code="", list_status="L", fields="ts_code")

    return all_stocks["ts_code"]


def get_index_stocks(ts_code: str, stocks_db_filename: str):
    data_df = read_from_db(f"index_info_{ts_code}".replace(".", "_"), stocks_db_filename, False, False)

    return data_df.con_code.unique()


def query_period_all_raw_data(ts_code: str, start_yr: int, end_yr: int, db_filename=None):
    if db_filename is not None:
        conn = sqlite3.connect(db_filename)

    raw_data_dict = {
        "daily_basic": [],
        "daily": [],
        "fina_indicator": [],
        "income": [],
        "balancesheet": [],
        "cashflow": []
    }
    interfaces = {
        "daily_basic": PRO.daily_basic,
        "daily": PRO.daily,
        "fina_indicator": PRO.fina_indicator,
        "income": PRO.income,
        "balancesheet": PRO.balancesheet,
        "cashflow": PRO.cashflow
    }

    for year_iter in range(start_yr, end_yr + 1):
        start_date = f"{year_iter}0101"
        end_date = f"{year_iter}1231"
        for func_iter_key in raw_data_dict:
            func_iter = interfaces[func_iter_key]
            data_df = func_iter(ts_code=ts_code, start_date=start_date, end_date=end_date)
            raw_data_dict[func_iter_key].append(data_df.fillna(np.nan))

    out_dict = {}
    for func_iter_key in raw_data_dict:
        out_dict[func_iter_key] = pd.concat(raw_data_dict[func_iter_key])

        if db_filename is not None:
            out_dict[func_iter_key].to_sql(f"{func_iter_key}_{ts_code}".replace(".", "_"), conn, if_exists="replace")

    if db_filename is not None:
        conn.close()

    return out_dict


def query_and_save_all_raw_data(db_filename: str, log_dir: str, start_yr=2018, end_yr=2023):
    timestamp = dt.today().strftime("%Y%m%d_%H%M%S")
    log_file = open(os.path.join(log_dir, f"{timestamp}.log"), "a")

    all_stocks = get_all_stocks()
    for ts_code in tqdm(all_stocks):
        try:
            query_period_all_raw_data(ts_code, start_yr, end_yr, db_filename)
        except Exception as err:
            print(f"Unsuccessful with error {err}: {ts_code}", file=log_file)
    log_file.close()


def __compute_daily_basic_period(df_daily_basic: pd.DataFrame):
    """
        Default: total_mv, pe, pb, turnover_rate
    """
    fields = "total_mv, pe, pb, turnover_rate, trade_date".split(", ")

    df_out = df_daily_basic[fields].set_index("trade_date")
    df_out.index = pd.to_datetime(df_out.index)
    df_out.sort_index(inplace=True)

    return df_out


def __compute_daily_period(df_daily_in: pd.DataFrame, **kwargs):
    """
    Default: reversal_rate, volatility
    kwargs: reversal_duration: int, vol_duration: int

    reversal_rate is the pct_chg over a period of time
    """
    reversal_duration = kwargs.get("reversal_duration", 5)
    vol_duration = kwargs.get("vol_duration", 30)
    df_daily = df_daily_in.set_index("trade_date")
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily.sort_index(inplace=True)

    df_out = pd.DataFrame(columns=["reversal_rate", "volatility"])
    df_out["reversal_rate"] = df_daily["close"].rolling(reversal_duration).apply(
        lambda arr: (arr.iloc[-1] - arr.iloc[0]) / arr.iloc[0] * 100
    )
    df_out["volatility"] = df_daily["pct_chg"].rolling(vol_duration).std()
    df_out = pd.concat([df_out, df_daily[["open", "high", "low", "close", "vol"]]], axis=1)

    return df_out


def __reindex_df(df: pd.DataFrame, df_index: pd.DataFrame):
    """
    df, df_index: After .set_index(.) and naming the index
    """
    df_index.index = pd.to_datetime(df_index.index)
    df.index = pd.to_datetime(df.index)
    df_index = df_index.sort_index()
    df = df.sort_index()
    mask = df.index.duplicated(keep="first")
    df = df[~mask]
    df = df.reindex(index=df_index.index, method="ffill")

    return df


def __compute_fina_indicator_period(df_fina_indicator: pd.DataFrame, df_daily: pd.DataFrame):
    """
    df_daily: for indices
    """
    fields = "roe, netprofit_yoy, or_yoy, assets_yoy, equity_yoy, end_date".split(", ")
    df_fina_indicator = df_fina_indicator[fields]
    df_daily = df_daily.set_index("trade_date")
    df_fina_indicator = df_fina_indicator.set_index("end_date")
    df_fina_indicator.index.name = "trade_date"
    df_fina_indicator = __reindex_df(df_fina_indicator, df_daily)

    return df_fina_indicator


def __compute_financial_statements_period(df_income: pd.DataFrame, df_bs: pd.DataFrame, df_cf: pd.DataFrame,
                                          df_daily: pd.DataFrame):
    # Reindexing
    df_dict = {
        "income": df_income,
        "bs": df_bs,
        "cf": df_cf
    }

    df_daily = df_daily.set_index("trade_date")

    for key in df_dict:
        df_iter = df_dict[key]
        df_iter = df_iter.set_index("end_date")
        df_iter.index.name = "trade_date"
        df_iter = __reindex_df(df_iter, df_daily)
        df_dict[key] = df_iter

    df_income = df_dict["income"]
    df_bs = df_dict["bs"]
    df_cf = df_dict["cf"]

    data_out = {
        "gross_profit_margin": df_income["total_profit"] / df_income["revenue"],
        "operating_profit_margin": df_income["operate_profit"] / df_income["revenue"],
        "net_profit_margin": df_income["n_income"] / df_income["revenue"],
        "operating_cash_flow_to_net_income": df_cf["n_cashflow_act"] / df_income["n_income"],
        "operating_cash_flow_to_revenue": df_cf["n_cashflow_act"] / df_income["revenue"],
        "current_ratio": df_bs["total_cur_assets"] / df_bs["total_cur_liab"],
        "cash_current_liability_ratio": df_cf["n_cashflow_act"] / df_bs["total_cur_liab"],
        "cash_liability_ratio": df_cf["n_cashflow_act"] / df_bs["total_liab"],
        "long_term_liability_operating_cash_flow_ratio": df_bs["lt_borr"] / df_cf["n_cashflow_act"]
    }

    df_out = pd.DataFrame(data_out)

    return df_out


def compute_factors_period(ts_code: str, db_raw_data_filename: str):
    """
    See compute_factors.ipynb for all factor names.
    """
    df_daily_basic = read_from_db(f"daily_basic_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)
    df_daily = read_from_db(f"daily_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)
    df_fina_indicator = read_from_db(f"fina_indicator_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)
    df_income = read_from_db(f"income_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)
    df_bs = read_from_db(f"balancesheet_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)
    df_cashflow = read_from_db(f"cashflow_{ts_code}".replace(".", "_"), db_raw_data_filename, False, False)

    df_out = list()
    df_out.append(__compute_daily_basic_period(df_daily_basic))
    df_out.append(__compute_daily_period(df_daily))
    df_out.append(__compute_fina_indicator_period(df_fina_indicator, df_daily))
    df_out.append(__compute_financial_statements_period(df_income, df_bs, df_cashflow, df_daily))
    df_out = pd.concat(df_out, axis=1)
    df_out["ts_code"] = ts_code

    return df_out.reset_index()


def compute_factors_and_save(ts_codes: Iterable[str], db_in_filename: str, db_out_filename: str, log_dir: str,
                             print_interval=50):
    conn = sqlite3.connect(db_out_filename)
    timestamp = dt.today().strftime("%Y%m%d_%H%M%S")
    log_file = open(os.path.join(log_dir, f"{timestamp}.log"), "a")
    orig_stdout = sys.stdout
    orig_stderr =  sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    all_factors = list()
    for i, ts_code in enumerate(tqdm(ts_codes)):
        try:
            df_out = compute_factors_period(ts_code, db_in_filename)
            all_factors.append(df_out)
            if i % print_interval == 1:
                print(pd.concat(all_factors, axis=0).set_index(["trade_date", "ts_code"]).sort_index())
                print("-" * 100)
        except Exception as err:
            print(f"Unsuccessful with error {err}: {ts_code}")

    all_factors = pd.concat(all_factors, axis=0).set_index(["trade_date", "ts_code"]).sort_index()
    all_factors.to_sql("factors_all_stocks", conn, if_exists="replace")

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()
    conn.close()

    return all_factors

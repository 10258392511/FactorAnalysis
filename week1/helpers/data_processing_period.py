import numpy as np
import pandas as pd
import sqlite3
import os
import sys

PATH = os.getcwd()
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

from FactorAnalysis.week1.helpers.data_processing import (
    remove_ST_and_list_date,
    remove_non_tradable_stocks,
    save_to_db,
)
from pandas.tseries.offsets import BDay
from FactorAnalysis.week1.helpers.utils import load_pro

pro = load_pro()
# hs300_start, hs300_end = "20240101", "20240131"
# zz500_start, zz500_end = "20231201", "20231231"
db_filename = "../data/stocks.db"


def construct_index(today: str):
    """
    For selected A share.
    """
    all_stocks = pro.stock_basic(ts_code="", list_status="L",
                                 fields="ts_code, symbol, name, list_status, list_date")
    trade_info = pro.daily("", trade_date=today)
    # hs300 = pro.index_weight(index_code="399300.SZ", start_date=hs300_start, end_date=hs300_end)
    # hs300.rename(columns={"con_code": "ts_code"}, inplace=True)
    # zz500 = pro.index_weight(index_code="000905.SH", start_date=zz500_start, end_date=zz500_end)
    # zz500.rename(columns={"con_code": "ts_code"}, inplace=True)

    all_stocks_out = remove_ST_and_list_date(all_stocks, min_list_duration=365)
    all_stocks_out_final = remove_non_tradable_stocks(all_stocks_out, trade_info)
    # hs300_final = remove_non_tradable_stocks(hs300, trade_info)
    # zz500_final = remove_non_tradable_stocks(zz500, trade_info)

    table_names = {
        "A_share": all_stocks_out_final,
        # "HS300": hs300_final,
        # "ZZ500": zz500_final
    }
    for table_name, stock_series in table_names.items():
        save_to_db(today, stock_series, table_name, db_filename)


def query_and_save_index_raw_table(index_ts_code: str, start_yr: int, end_yr: int, db_filename=None):
    data_dict = {
        "index_info": [],
    }
    for year_iter in range(start_yr, end_yr + 1):
        start_date = f"{year_iter}0101"
        end_date = f"{year_iter}1231"
        data_dict["index_info"].append(pro.index_weight(index_code=index_ts_code,
                                                        start_date=start_date, end_date=end_date))

    out_dict = {}
    if db_filename is not None:
        conn = sqlite3.connect(db_filename)

    for key in data_dict:
        out_dict[key] = pd.concat(data_dict[key])
        if db_filename is not None:
            out_dict[key].to_sql(f"{key}_{index_ts_code}".replace(".", "_"), conn, if_exists="replace")

    if db_filename is not None:
        conn.close()

    return out_dict


def main(start_date: str, end_date: str, print_interval=10):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    today_date = start_date

    counter = 0
    while today_date <= end_date:
        today = today_date.strftime("%Y%m%d")
        if counter % print_interval == 0:
            print(f"Current: {today}")

        construct_index(today)
        today_date = today_date + BDay(1)
        counter += 1


if __name__ == "__main__":
    main(start_date="20180101", end_date="20231231")

import numpy as np
import pandas as pd
import sqlite3

from datetime import datetime as dt
from typing import Union, Iterable


def remove_ST_and_list_date(stocks: pd.DataFrame, min_list_duration=365) -> pd.DataFrame:
    """
    stocks: withh columns "name", "list_date"
    """
    today = pd.to_datetime(dt.today().strftime("%Y%m%d"))
    mask_ST = stocks.name.str.contains("ST")
    list_duration = (today - pd.to_datetime(stocks.list_date)) / pd.Timedelta(days=min_list_duration)
    mask_long_duration = (list_duration > 1)
    stocks_out = stocks[(~mask_ST) & mask_long_duration]

    return stocks_out


def remove_non_tradable_stocks(stocks: pd.DataFrame, trade_info: pd.DataFrame) -> pd.Series:
    """
    stocks: ts_code
    trade_info: ts_code, vol

    Returns
    -------
    A pd.Series of stock code
    """
    trade_info = trade_info[trade_info.vol > 0]
    stocks_out = trade_info.merge(stocks, on="ts_code", how="inner")

    return stocks_out.ts_code


def read_table_names(db_filename: str):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    stmt = """
    SELECT name FROM sqlite_master 
    WHERE type="table"
    """
    cursor.execute(stmt)
    res = cursor.fetchall()
    conn.close()

    return res


def remove_table(db_filename: str, table_name: str):
    all_table_names = read_table_names(db_filename)
    if (table_name,) not in all_table_names:
        return
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    stmt = f"""
    DROP TABLE {table_name}
    """
    cursor.execute(stmt)
    conn.commit()
    conn.close()


def save_to_db(today: str, selected_stock_code: Iterable,
               table_name: str, db_filename: str):
    """
    date, ts_code | status

    status: 1
    """
    # if isinstance(today, str):
    #     today = pd.to_datetime(today)

    all_table_names = read_table_names(db_filename)
    if (table_name,) in all_table_names:
        df_prev = read_from_db(table_name, db_filename, parse_dates=False)
        if today in df_prev.index.get_level_values(0).unique():
            return

    df = pd.DataFrame(columns=selected_stock_code)
    # df.loc[today] = 0
    df.loc[today, selected_stock_code] = 1
    df = df.stack(level=-1).to_frame()
    df.rename(columns={0: "status"}, inplace=True)

    # df_prev = read_from_db(table_name, db_filename, parse_dates=False)
    # if len(df_prev) > 0:
    #     df = pd.concat([df_prev, df], axis=0)
    # print(df)

    conn = sqlite3.connect(db_filename)
    df.to_sql(table_name, conn, if_exists="append", index=True, index_label=["date", "ts_code"])
    conn.close()


def read_from_db(table_name: str, db_filename: str, parse_dates=True, use_index=True):
    conn = sqlite3.connect(db_filename)
    stmt = f"""
    SELECT *
    FROM {table_name}
    """
    if use_index:
        df = pd.read_sql(stmt, conn)
        date_colname = None
        if "date" in df.columns:
            date_colname = "date"
        elif "trade_date" in df.columns:
            date_colname = "trade_date"
        if parse_dates:
            df = pd.read_sql(stmt, conn, index_col=[date_colname, "ts_code"], parse_dates=[date_colname])
        else:
            df = pd.read_sql(stmt, conn, index_col=[date_colname, "ts_code"])
    else:
        df = pd.read_sql(stmt, conn)
    conn.close()

    df.fillna(np.nan, inplace=True)
    df.sort_index(inplace=True)

    return df

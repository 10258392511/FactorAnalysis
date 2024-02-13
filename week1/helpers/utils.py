import pandas as pd
import tushare as ts
import sqlite3
import yaml
import re
import os
import glob

ROOT = os.path.abspath(__file__)
for _ in range(3):
    ROOT = os.path.dirname(ROOT)

from FactorAnalysis.week1.helpers.data_processing import read_table_names, read_from_db
from zipline import get_calendar
from tqdm import tqdm


def load_pro(env_filename=None):
    if env_filename is None:
        env_filename = os.path.join(ROOT, "configs/environ.yml")
    with open(env_filename, "r") as rf:
        token = yaml.safe_load(rf)["token"]

    pro = ts.pro_api(token)

    return pro


def save_to_db(df: pd.DataFrame, db_filename: str, table_name: str, if_exists="replace"):
    conn = sqlite3.connect(db_filename)
    df.to_sql(table_name, conn, if_exists=if_exists)
    conn.close()


def save_price_as_csv(db_in_filename: str, out_dir: str):
    table_names = read_table_names(db_in_filename)
    table_names = [name[0] for name in table_names if "daily" in name[0] and "basic" not in name[0]]

    columns = ["trade_date", "open", "high", "low", "close", "vol"]
    pattern = r"(\d+)_(S[HZ]|BJ)"
    pbar = tqdm(table_names)
    for table_name in pbar:
        price_df = read_from_db(table_name, db_in_filename)
        price_df_converted = price_df.reset_index()[columns].rename(columns={
            "trade_date": "date",
            "vol": "volume"
        }).set_index("date")
        out_filename = "".join(re.findall(pattern, table_name)[0])  # e.g. [("000001", "SZ")] -> 000001SZ
        price_df_converted.to_csv(os.path.join(out_dir, f"{out_filename}.csv"))
        pbar.set_description(f"{out_filename}")


def impute_for_all_trading_days(csv_in_dir: str, calendar_name: str, csv_out_dir: str,
                                start="20180101", end="20231231", period="1D"):
    all_csvs = glob.glob(os.path.join(csv_in_dir, "*.csv"))
    calendar = get_calendar(calendar_name)
    trading_index = calendar.trading_index(start, end, period)

    pbar = tqdm(all_csvs)
    for filename in pbar:
        pbar.set_description(os.path.basename(filename))
        df = pd.read_csv(filename, index_col="date", parse_dates=True)
        df = df.reindex(trading_index).ffill().fillna(0.)
        df.to_csv(os.path.join(csv_out_dir, os.path.basename(filename)))

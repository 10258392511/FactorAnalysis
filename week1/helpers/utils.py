import pandas as pd
import tushare as ts
import sqlite3
import yaml
import os

ROOT = os.path.abspath(__file__)
for _ in range(3):
    ROOT = os.path.dirname(ROOT)


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


import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(4):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

from FactorAnalysis.week1.helpers.utils import save_to_db


def remove_duplicates(factors: pd.DataFrame):

    return factors[~factors.index.duplicated(keep="first")]


def pop_OHLCV(factors: pd.DataFrame, start_date=None):
    # factors: index: trade_date, ts_code
    factors = remove_duplicates(factors)
    if start_date is not None:
        factors = factors.loc[(slice(start_date, None),)]
    columns = ["open", "high", "low", "close", "vol"]
    OHLCV = factors[columns]
    factors = factors.drop(columns=columns)

    return OHLCV, factors


def __apply_func(factors: pd.DataFrame, func, level=0):
    # factors: index: trade_date, ts_code
    factors = (
        factors
        .groupby(level=level, as_index=False)
        .apply(func)
        .reset_index()
        .drop(columns="level_0")
        .set_index(["trade_date", "ts_code"])
    )

    return factors


def __imputation(factors: pd.DataFrame, start_date=None, end_date=None):
    # factors: index: trade_date, ts_code
    if start_date is None:
        start_date = pd.to_datetime("2019-01-01")
        end_date = factors.index.get_level_values(0).unique()[-1]
    factors = factors.loc[(slice(start_date, end_date),), :]

    # Remove rows with duplicate indices
    factors = remove_duplicates(factors)

    # ffill(.) in time for each ts_code (i.e. group by ts_code)
    factors = __apply_func(factors, lambda df: df.ffill(), level=1)

    # fillna(.) in cross-section for each date with median (i.e. group by trade_date)
    factors = __apply_func(factors, lambda df: df.fillna(df.median()))

    # Drop any remaining rows with NaN
    factors.dropna(inplace=True)

    return factors


def __MAD_winsorization(factors: pd.DataFrame, multiple=2.5):
    # factors: index: trade_date, ts_code
    def func(df: pd.DataFrame):
        median = np.median(df.values, axis=0)  # (D,)
        dev = df.values - median  # (N, D)
        dev_abs = np.abs(dev)  # (N, D)
        dev_mul = np.median(dev_abs, axis=0) * multiple  # (D,)
        dev = dev.clip(-dev_mul, dev_mul)
        out = median + dev
        df_out = pd.DataFrame(out, index=df.index, columns=df.columns)

        return df_out

    factors = __apply_func(factors, func)

    return factors


def __standarization(factors: pd.DataFrame):
    # factors: index: trade_date, ts_code
    def func(df: pd.DataFrame):
        df_mean = df.mean()
        df_std = df.std()

        return (df - df_mean) / df_std

    factors = __apply_func(factors, func)

    return factors


def check_mean_std(factors: pd.DataFrame):
    # factors: index: trade_date, ts_code
    factors_mean = (
        factors
        .groupby(level=0)
        .mean()
    )
    factors_std = (
        factors
        .groupby(level=0)
        .std()
    )
    return factors_mean, factors_std


def __neutralization_mv(factors: pd.DataFrame):
    # factors: index: trade_date, ts_code
    def func(df: pd.DataFrame):
        dep = np.log(df["total_mv"])

        def func_on_df(col: pd.Series):
            indep = sm.add_constant(col)
            model = sm.OLS(dep, indep)
            res = model.fit()

            return res.resid

        df_out = df.apply(func_on_df)

        return df_out

    # Reg in time: for each ts_code: factor ~ log(total_mv) (i.e. group by ts_code)
    factors = __apply_func(factors, func, level=1)

    return factors


def count_nan(df: pd.DataFrame):
    nans = df.isna().sum()

    return nans / df.shape[0]


def data_processing_pipeline(factors: pd.DataFrame, return_intermediates=False, db_save_filename=None):
    ohlcv, factors_all_no_prices = pop_OHLCV(factors)
    print("Imputing...")
    factors_imputed = __imputation(factors_all_no_prices)
    print("Winsorizing...")
    factors_winsorized = __MAD_winsorization(factors_imputed)
    print("Neutralizing...")
    factors_neutralized = __neutralization_mv(factors_winsorized)
    print("Normalizing...")
    factors_norm = __standarization(factors_neutralized)

    if return_intermediates:
        factors_out = [
            factors_all_no_prices,
            factors_imputed,
            factors_winsorized,
            factors_neutralized,
            factors_norm
        ]
    else:
        factors_out = factors_norm

    if db_save_filename is not None:
        print("Saving to DB...")
        save_to_db(factors_norm, db_save_filename, "factors_all_stocks")

    return ohlcv, factors_out

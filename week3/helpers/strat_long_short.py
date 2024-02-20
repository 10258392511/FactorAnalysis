import numpy as np
import pandas as pd
import pyfolio as pf
import os

from .strat_load_factors import (
    load_combined_factors_quantile,
    load_mv,
    SignalData,
    QuantileFactor,
    MVFactor
)
from zipline.algorithm import TradingAlgorithm
from zipline.protocol import BarData
from zipline import run_algorithm
from zipline.api import (
    set_commission,
    set_slippage,
    attach_pipeline,
    pipeline_output,
    schedule_function,
    order_target_percent
)
from zipline.finance.commission import PerShare
from zipline.finance.slippage import FixedSlippage
from zipline.pipeline import Pipeline
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.utils.events import date_rules, time_rules
from typing import Union


def initialize(context: TradingAlgorithm):
    set_commission(PerShare(0., 0.))
    set_slippage(FixedSlippage(spread=0.))

    schedule_function(rebalance, date_rules.every_day(), time_rules.market_close())

    attach_pipeline(make_pipeline(), "my_pipeline")

    context.num_days = 0
    context.rebalance_period = 5
    context.num_quantiles = 5
    context.print_interval = 100


def make_pipeline():
    quantiles = QuantileFactor()
    pipeline = Pipeline(
        columns={
            "quantile": quantiles,
            "mv": MVFactor()
        },
        screen=~quantiles.isnan()
    )

    return pipeline


def before_trading_start(context: TradingAlgorithm, data: BarData):
    # print(context.portfolio.positions)
    context.pipeline_data = pipeline_output("my_pipeline")
    context.num_days += 1


def rebalance(context: TradingAlgorithm, data: BarData):
    """
    Long positions only
    """
    if context.num_days % context.rebalance_period != 0:
        return

    if context.num_days % context.print_interval == 0:
        print(context.get_datetime())
    df = context.pipeline_data  # equity_id | quantile, mv
    q_mask = df["quantile"] == context.num_quantiles - 1
    # q_mask = df["quantile"] == 0

    df = df[q_mask].copy()
    assets = set(df.index)
    df["weight"] = df["mv"] / df["mv"].sum()

    for asset_iter in context.portfolio.positions:
        if asset_iter not in assets:
            order_target_percent(asset_iter, 0)

    for asset_iter in assets:
        order_target_percent(asset_iter, df.loc[asset_iter, "weight"])


def run_algo(
        combined_factors: pd.DataFrame,
        factors_all: pd.DataFrame,
        weighting_name: str,
        save_dir=None,
        **kwargs
):
    """
    combined_factors: all weighting types
    kwargs: start_date, end_date, capital_base
    """
    factors_quantile, _ = load_combined_factors_quantile(combined_factors, weighting_name)
    factors_mv, _ = load_mv(factors_all)
    signal_loader = {
        SignalData.quantiles: DataFrameLoader(SignalData.quantiles, factors_quantile),
        SignalData.mvs: DataFrameLoader(SignalData.mvs, factors_mv)
    }

    start_date = pd.to_datetime(kwargs.get("start_date", "20190102"))
    end_date = pd.to_datetime(kwargs.get("end_date", "20231231"))
    capital_base = kwargs.get("capital_base", 1e6)
    res_df = run_algorithm(
        start=start_date,
        end=end_date,
        capital_base=capital_base,
        initialize=initialize,
        before_trading_start=before_trading_start,
        bundle="AShareBundle",
        custom_loader=signal_loader
    )

    if save_dir is not None:
        filename = os.path.join(save_dir, f"{weighting_name}.pkl")
        res_df.to_pickle(filename)

    return res_df


def load_benchmark(benchmark_filename: str):
    assert ".csv" in benchmark_filename
    benchmark_df = pd.read_csv(benchmark_filename, usecols=["trade_date", "pct_chg"], index_col="trade_date",
                               parse_dates=True).sort_index().tz_localize("UTC")
    benchmark_df.rename(columns={"pct_chg": "benchmark"}, inplace=True)

    return benchmark_df["benchmark"] / 100


def create_full_tear_sheet(res_df: Union[pd.DataFrame, str], benchmark_filename: str):
    if isinstance(res_df, str):
        assert ".pkl" in res_df
        res_df = pd.read_pickle(res_df)

    benchmark_series = load_benchmark(benchmark_filename)
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(res_df)

    benchmark_series.index = benchmark_series.index.normalize()
    res_df.index = res_df.index.normalize()
    benchmark_series = benchmark_series.to_frame().reindex(res_df.index).ffill()
    benchmark_series = benchmark_series.iloc[:, 0]
    pf.create_full_tear_sheet(returns, positions, transactions, benchmark_rets=benchmark_series, round_trips=True)

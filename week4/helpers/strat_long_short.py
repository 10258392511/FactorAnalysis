import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(4):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)


from FactorAnalysis.week3.helpers.strat_load_factors import (
    load_combined_factors_quantile,
    load_mv,
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
from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.domain import US_EQUITIES
from zipline.finance.commission import PerShare
from zipline.finance.slippage import FixedSlippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.utils.events import date_rules, time_rules
from typing import Union


class SignalData(DataSet):
    quantiles_long = Column(dtype=float)
    quantiles_short = Column(dtype=float)
    mvs = Column(dtype=float)
    domain = US_EQUITIES


class QuantileLongFactor(CustomFactor):
    inputs = [SignalData.quantiles_long]
    window_length = 1

    def compute(self, today, assets: np.ndarray, out: np.ndarray, quantiles: np.ndarray):
        out[:] = quantiles


class QuantileShortFactor(CustomFactor):
    inputs = [SignalData.quantiles_short]
    window_length = 1

    def compute(self, today, assets: np.ndarray, out: np.ndarray, quantiles: np.ndarray):
        out[:] = quantiles


class MVFactor(CustomFactor):
    inputs = [SignalData.mvs]
    window_length = 1

    def compute(self, today, assets: np.ndarray, out: np.ndarray, mvs: np.ndarray):
        out[:] = mvs


def initialize(context: TradingAlgorithm):
    set_commission(PerShare(0., 0.))
    set_slippage(FixedSlippage(spread=0.))

    schedule_function(rebalance, date_rules.every_day(), time_rules.market_close())

    attach_pipeline(make_pipeline(), "my_pipeline")

    context.num_days = 0
    context.rebalance_period = 5
    context.num_quantiles = 5
    context.print_interval = 20


def make_pipeline():
    quantiles_long = QuantileLongFactor()
    quantiles_short = QuantileShortFactor()
    pipeline = Pipeline(
        columns={
            "quantile_long": quantiles_long,
            "quantile_short": quantiles_short,
            "mv": MVFactor()
        },
        screen=~(quantiles_long.isnan() | quantiles_short.isnan())
    )

    return pipeline


def before_trading_start(context: TradingAlgorithm, data: BarData):
    # print(context.portfolio.positions)
    context.pipeline_data = pipeline_output("my_pipeline")
    context.num_days += 1


def rebalance(context: TradingAlgorithm, data: BarData):
    if context.num_days % context.rebalance_period != 0:
        return

    if context.num_days % context.print_interval == 0:
        print(context.get_datetime())

    df = context.pipeline_data  # equity_id | quantile_long, quantile_short, mv
    q_long_mask = df["quantile_long"] == 4
    q_short_mask = df["quantile_short"] == 0
    q_mask = q_long_mask | q_short_mask

    df = df[q_mask].copy()
    assets_long = set(df[q_long_mask].index)
    assets_short = set(df[q_short_mask].index)
    assets_short = assets_short.difference(assets_long)
    assets = assets_long.union(assets_short)
    df["weight"] = df["mv"] / df["mv"].sum()

    for asset_iter in context.portfolio.positions:
        if asset_iter not in assets:
            order_target_percent(asset_iter, 0)

    for asset_iter in assets_long:
        order_target_percent(asset_iter, df.loc[asset_iter, "weight"])

    for asset_iter in assets_short:
        order_target_percent(asset_iter, df.loc[asset_iter, "weight"])


def run_algo(
        combined_factors: pd.DataFrame,
        factors_all: pd.DataFrame,
        save_dir=None,
        **kwargs
):
    """
    combined_factors: all weighting types
    kwargs: start_date, end_date, capital_base
    """
    factors_quantile_long, _ = load_combined_factors_quantile(combined_factors, "IR_5D")
    factors_quantile_short, _ = load_combined_factors_quantile(combined_factors, "IC_rank_1D")
    factors_mv, _ = load_mv(factors_all)
    signal_loader = {
        SignalData.quantiles_long: DataFrameLoader(SignalData.quantiles_long, factors_quantile_long),
        SignalData.quantiles_short: DataFrameLoader(SignalData.quantiles_short, factors_quantile_short),
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
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, "backtest.pkl")
        res_df.to_pickle(filename)

    return res_df

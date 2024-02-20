import numpy as np
import pandas as pd

from zipline import TradingAlgorithm
from zipline.protocol import BarData
from zipline.data import bundles
from zipline.api import (
    pipeline_output,
    set_slippage,
    set_commission,
    attach_pipeline
)
from zipline.finance.commission import PerShare
from zipline.finance.slippage import FixedSlippage
from zipline.pipeline import (
    Pipeline,
    CustomFactor
)
from zipline.pipeline.data import DataSet, Column
from zipline.pipeline.domain import US_EQUITIES


def load_combined_factors_quantile(combined_factors: pd.DataFrame, weighting_name: str, quantiles=5):
    """
    combined_factors: trade_date, ts_code | weighting_name1, ...
    """
    bundle_data = bundles.load("A-share-csvdir-bundle")

    factors = combined_factors[weighting_name].unstack(level=-1)  # trade_date | ts_code1, ...
    colname_dict = {colname: colname.replace(".", "") for colname in factors.columns}
    assets = bundle_data.asset_finder.lookup_symbols(colname_dict.values(), as_of_date=None)
    assets_id = [asset.sid for asset in assets]
    colname_dict = dict(zip(colname_dict.keys(), assets_id))
    factors.rename(columns=colname_dict, inplace=True)
    factors = factors.apply(lambda row: pd.qcut(row, quantiles, labels=False, duplicates="drop", precision=6), axis=1)

    return factors, assets


def load_mv(factors_raw: pd.DataFrame):
    """
    factors_raw: trade_date, ts_code | weighting_name1, ...
    """
    bundle_data = bundles.load("A-share-csvdir-bundle")

    factors = factors_raw["total_mv"].unstack(level=-1)  # trade_date | ts_code1, ...
    colname_dict = {colname: colname.replace(".", "") for colname in factors.columns}
    assets = bundle_data.asset_finder.lookup_symbols(colname_dict.values(), as_of_date=None)
    assets_id = [asset.sid for asset in assets]
    colname_dict = dict(zip(colname_dict.keys(), assets_id))
    factors.rename(columns=colname_dict, inplace=True)
    # factors /= np.nansum(factors.values, axis=1, keepdims=True)

    return factors, assets


class SignalData(DataSet):
    quantiles = Column(dtype=float)
    mvs = Column(dtype=float)
    domain = US_EQUITIES


class QuantileFactor(CustomFactor):
    inputs = [SignalData.quantiles]
    window_length = 1

    def compute(self, today, assets: np.ndarray, out: np.ndarray, quantiles: np.ndarray):
        out[:] = quantiles


class MVFactor(CustomFactor):
    inputs = [SignalData.mvs]
    window_length = 1

    def compute(self, today, assets: np.ndarray, out: np.ndarray, mvs: np.ndarray):
        out[:] = mvs


def initialize(context: TradingAlgorithm):
    set_commission((PerShare(0., 0.)))
    set_slippage(FixedSlippage(spread=0.))
    attach_pipeline(make_pipeline(), "my_pipeline")


def make_pipeline():
    quantiles = QuantileFactor()
    return Pipeline(
        columns={
            "quantile": quantiles,
            "mv": MVFactor()
        },
        screen=~quantiles.isnan()
    )


def before_trading_start(context: TradingAlgorithm, data: BarData):
    context.pipeline_data = pipeline_output("my_pipeline")
    print(context.pipeline_data)


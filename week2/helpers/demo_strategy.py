from zipline import TradingAlgorithm
from zipline.protocol import BarData
from zipline.api import (
    set_slippage,
    set_commission,
    schedule_function,
    attach_pipeline,
    pipeline_output,
    record
)
from zipline.finance.commission import PerShare
from zipline.finance.slippage import FixedSlippage
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns
from zipline.utils.events import date_rules, time_rules


def initialize(context: TradingAlgorithm):
    set_commission(PerShare(0, 0))
    set_slippage(FixedSlippage(0.))

    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

    attach_pipeline(make_pipeline(), "my_pipeline")

    context.num_days = 1


def make_pipeline():
    columns = {
        "returns": Returns(window_length=3),
        "stock_price": USEquityPricing.close.latest
    }

    return Pipeline(columns=columns)


def before_trading_start(context: TradingAlgorithm, data: BarData):
    print(context.get_datetime())
    context.pipeline_data = pipeline_output("my_pipeline")
    if context.num_days < 5 or context.num_days > 1500:  # 20180101 to 20231231
        print(context.pipeline_data)
    context.num_days += 1


def record_vars(context: TradingAlgorithm, data: BarData):
    record(
        returns=context.pipeline_data["returns"].mean(),
        stock_price=context.pipeline_data["stock_price"].mean()
    )

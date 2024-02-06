import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import alphalens as al

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def get_clean_factor_and_forward_returns(price: pd.DataFrame, factor: pd.Series, periods=(1, 5, 10),
                                         quantiles=5) -> pd.DataFrame:
    """
    price (ohlcv), factor (one factor, Series): short form; index: trade_date, ts_code
    """
    price.index.names = ["date", "asset"]
    factor.index.names = ["date", "asset"]
    price = price["open"].unstack("asset")  # long form: date | asset1, ...
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor,
        price,
        quantiles=quantiles,
        bins=None
    )

    return factor_data


def compute_ic(factor_data: pd.DataFrame, rank=True) -> pd.DataFrame:
    ic_df = pd.DataFrame(columns=factor_data.columns[:-2])  # e.g. 1D, 5D, 10D
    for col in ic_df.columns:
        if not rank:
            def func(df: pd.DataFrame):
                return pearsonr(df[col], df["factor"]).statistic
        else:
            def func(df: pd.DataFrame):
                return spearmanr(df[col], df["factor"]).statistic
        ic_df[col] = (
            factor_data
            .groupby("date")
            .apply(func)
        )

    # e.g. date | 1D, 5D, 10D
    return ic_df


def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    Source: https://github.com/stefan-jansen/alphalens-reloaded/blob/d4490ba1290f1f135ed398d1b3601569e0e7996b/src/alphalens/plotting.py#L696
    Changed to show colorbar.

    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month], names=["year", "month"]
    )

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.items()):
        sns.heatmap(
            ic.unstack(),
            annot=False,
            alpha=1.0,
            center=0.0,
            # annot_kws={"size": 7},
            linewidths=0.01,
            linecolor="white",
            cmap=cm.coolwarm_r,
            cbar=True,
            ax=a,
        )
        a.set(ylabel="", xlabel="")

        a.set_title("Monthly Mean {} Period IC".format(periods_num))

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def create_ic_plots(ic_df: pd.DataFrame, by_time="1M"):
    """
    ic_df: date | 1D, 5D, 10D
    by_time: for plot_monthly_ic_heatmap(.)
    """
    plot_funcs = [al.plotting.plot_ic_ts, al.plotting.plot_ic_hist, al.plotting.plot_ic_qq]
    axes = []
    for plot_func in plot_funcs:
        axes.append(plot_func(ic_df))
    mean_ic = ic_df.resample(rule=by_time).mean()
    axes.append(plot_monthly_ic_heatmap(mean_ic))
    plt.show()

    return axes


def factor_analysis_single(price: pd.DataFrame, factor: pd.Series, **kwargs):
    """
    kwargs: periods, quantiles, by_time, if_plot

    Returns
    -------
    ic_df, ic_r_df: date | period1, ...
    ir_series, ir_r_series: (e.g. 1D) | (val: Series)
    """
    periods = kwargs.get("period", (1, 5, 10))
    quantiles = kwargs.get("quantiles", 5)
    by_time = kwargs.get("by_time", "1M")
    if_plot = kwargs.get("if_plot", True)

    factor_data = get_clean_factor_and_forward_returns(price, factor, periods=periods, quantiles=quantiles)
    ic_df = compute_ic(factor_data, False)
    ic_r_df = compute_ic(factor_data, True)
    if if_plot:
        print("IC plots:")
        create_ic_plots(ic_df, by_time=by_time)
        print("-" * 100)
        print("Rank IC plots:")
        create_ic_plots(ic_r_df, by_time=by_time)

    ir_series = ic_df.mean() / ic_df.std()
    ir_r_series = ic_r_df.mean() / ic_r_df.std()

    return ic_df, ic_r_df, ir_series, ir_r_series


def factor_analysis_all(price: pd.DataFrame, factors: pd.DataFrame, **kwargs):
    """
    price (ohlcv), factor: short form; trade_date, ts_code | factor1, ...
    kwargs: See factor_analysis_single(.)

    Returns
    -------
    all_ic_df, all_ic_r_df: date, period | factor1, ...
    all_ir_df, all_ir_r_df: period | factor1, ...
    """
    all_ic_df = pd.DataFrame(columns=factors.columns)
    all_ic_r_df = pd.DataFrame(columns=factors.columns)
    all_ir_df = pd.DataFrame(columns=factors.columns)
    all_ir_r_df = pd.DataFrame(columns=factors.columns)

    for col in tqdm(factors.columns):
        print(f"Factor {col}")
        ic_df, ic_r_df, ir_series, ir_r_series = factor_analysis_single(price, factors[col], **kwargs)
        all_ic_df[col] = ic_df.stack(level=-1)  # date, period | factor1, ...
        all_ic_r_df[col] = ic_r_df.stack(level=-1)
        all_ir_df[col] = ir_series
        all_ir_r_df[col] = ir_r_series
        print("=" * 100)

    return all_ic_df, all_ic_r_df, all_ir_df, all_ir_r_df

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.linear_model import LinearRegression

from trot.config import Config


def make_multiclass_parity_plot(
    cfg: Config,
    df: pl.DataFrame,
    y_col: str,
    units: str = "eV",
) -> None:
    plt.figure()
    df = df.unpivot(index=[y_col]).with_columns(pl.col("variable"))
    sns.lmplot(data=df, x="value", y=y_col, hue="variable", legend=False)
    min_val = min(df["value"].min(), df[y_col].min())
    max_val = max(df["value"].max(), df["value"].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.legend(title="Experts")
    plt.title(label="Parity Plot with std_factor = 1", fontsize=20)
    plt.xlabel(xlabel=f"ML ({units})", fontsize=16)
    plt.ylabel(ylabel=f"{y_col} ({units})", fontsize=16, rotation=0, labelpad=36)
    plt.savefig(cfg.paths.results.iter_avg_parity_plot, bbox_inches="tight")
    plt.close()


def make_parity_plot(
    cfg: Config,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
):
    plt.figure()
    # Plot parity line
    x_grid = np.linspace(min(x_axis), max(x_axis), 100)
    plt.plot(x_grid, x_grid, color="black", linewidth=2, linestyle="--")
    # Plot data and trendline
    model = LinearRegression().fit(x_axis, y_axis)
    plt.plot(x_grid, model.predict(x_grid), color="blue", linewidth=2)
    plt.scatter(x_axis, y_axis, color="blue", alpha=0.7)
    plt.title(label=title, fontsize=20)
    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.ylabel(ylabel=y_label, fontsize=16, rotation=0, labelpad=36)
    plt.savefig(cfg.paths.results.adjusted_parity_plot, bbox_inches="tight")
    plt.close()


def make_bar_plot(
    cfg: Config,
    x_axis: list,
    y_axis: list,
    x_label: str,
    y_label: str,
    title: str,
):
    plt.figure()
    plt.bar(x_axis, y_axis)
    plt.ylim(0, max(y_axis) * 1.2)
    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.xticks(np.arange(min(x_axis), max(x_axis) + 1, 1))
    plt.ylabel(ylabel=y_label, fontsize=16, rotation=0, labelpad=40)
    plt.title(label=title)
    plt.savefig(cfg.paths.results.bar_plot, bbox_inches="tight")
    plt.close()


def make_histogram_plot(
    cfg: Config,
    data: np.ndarray,
    x_label: str,
    bins: int = 5,
) -> None:
    plt.figure()
    plt.hist(x=data, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.ylabel("Frequency", fontsize=16, rotation=0, labelpad=40)
    plt.title(label=f"Histogram: Bins = {bins}")
    plt.savefig(cfg.paths.results.hist_plot, bbox_inches="tight")
    plt.close()

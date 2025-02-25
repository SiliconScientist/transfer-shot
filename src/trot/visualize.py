import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import re

from trot.config import Config


def clean_column_name(name):
    match = re.match(r"^([A-Za-z0-9]+)", name)
    return match.group(1).upper() if match else name


def make_multiclass_parity_plot(
    cfg: Config,
    df: pl.DataFrame,
    y_col: str,
    units: str = "eV",
) -> None:
    df = df.unpivot(index=[y_col]).with_columns(pl.col("variable"))
    sns.lmplot(data=df, x=y_col, y="value", hue="variable", legend=False)
    min_val = min(df[y_col].min(), df["value"].min())
    max_val = max(df[y_col].max(), df[y_col].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linewidth=2,
        linestyle="--",
    )
    plt.legend(title="Experts")
    plt.title(label="Parity Plot with std_factor = 1", fontsize=20)
    plt.xlabel(xlabel=f"{y_col} ({units})", fontsize=16)
    plt.ylabel(ylabel=f"ML ({units})", fontsize=16, rotation=0, labelpad=36)
    plt.savefig(cfg.paths.results.parity_plot, bbox_inches="tight")


def make_parity_plot(
    cfg: Config,
    model,
    parity_line: np.ndarray,
    x_axis: list,
    y_axis: list,
    x_label: str,
    y_label: str,
    title: str,
):
    plt.plot(parity_line, parity_line, color="black", linewidth=2, linestyle="--")
    plt.plot(parity_line, y_axis, color="red", linewidth=2)
    plt.legend(title="Experts")
    plt.title(label=title, fontsize=20)
    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.ylabel(ylabel=y_label, fontsize=16, rotation=0, labelpad=36)
    plt.savefig(cfg.paths.results.parity_plot, bbox_inches="tight")


def make_bar_plot(
    cfg: Config,
    x_axis: list,
    y_axis: list,
    x_label: str,
    y_label: str,
    title: str,
):
    plt.figure(figsize=(8, 5))
    plt.bar(x_axis, y_axis)
    plt.xlabel(xlabel=x_label, fontsize=16)
    plt.xticks(np.arange(min(x_axis), max(x_axis) + 1, 1))
    plt.ylabel(ylabel=y_label, fontsize=16, rotation=0, labelpad=36)
    plt.title(label=title)
    plt.savefig(cfg.paths.results.bar_plot, bbox_inches="tight")

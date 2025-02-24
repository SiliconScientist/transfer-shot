import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import re


def clean_column_name(name):
    match = re.match(r"^([A-Za-z0-9]+)", name)
    return match.group(1).upper() if match else name


def make_parity_plot(df: pl.DataFrame, y_col: str, units: str = "eV") -> None:
    df = df.unpivot(index=[y_col]).with_columns(
        pl.col("variable")
    )  # .str.replace("_energy", "").str.to_uppercase())
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
    plt.xlabel(xlabel=f"{y_col} ({units})", fontsize=16)
    plt.ylabel(ylabel=f"ML ({units})", fontsize=16, rotation=0, labelpad=36)
    plt.savefig("data/results/visualizations/parity_plot.png", bbox_inches="tight")

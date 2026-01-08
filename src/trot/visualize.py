from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.linear_model import LinearRegression
from typing import Union

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
    plt.xlabel(xlabel=f"ML ({units})", fontsize=16)
    plt.ylabel(ylabel=f"{y_col} ({units})", fontsize=16, rotation=0, labelpad=36)
    plt.savefig(cfg.paths.results.multiclass_parity_plot, bbox_inches="tight")
    plt.close()


def make_parity_plot(
    cfg: Config,
    x_axis: np.ndarray,
    yerr: np.ndarray,
    y_axis: np.ndarray,
    x_label: str,
    y_label: str,
    inset: Union[str, None] = None,
    fontsize: int = 18,
):
    plt.figure()
    # Plot parity line
    x_grid = np.linspace(min(x_axis), max(x_axis), 100)
    plt.plot(x_grid, x_grid, color="black", linewidth=2, linestyle="--")
    # Plot data and trendline
    model = LinearRegression().fit(x_axis, y_axis)
    plt.plot(x_grid, model.predict(x_grid), color="blue", linewidth=2)
    plt.scatter(x_axis, y_axis, color="blue", alpha=0.7, label="Data Points")
    # plt.errorbar(
    #     x_axis,
    #     y_axis,
    #     yerr=yerr,
    #     color="blue",
    #     fmt="o",
    #     capsize=5,
    #     alpha=0.7,
    # )
    plt.text(
        x=max(x_axis),
        y=min(y_axis),
        s=inset,
        fontsize=fontsize,
        color="black",
        ha="right",
        va="bottom",
        multialignment="center",
    )
    plt.xlabel(xlabel=x_label, fontsize=fontsize)
    plt.ylabel(ylabel=y_label, fontsize=fontsize, labelpad=10)
    plt.savefig(cfg.paths.results.parity_plot, bbox_inches="tight")
    plt.close()


def make_bar_plot(
    cfg: Config,
    x_axis: list,
    y_axis: list,
    yerr: list,
    x_label: str,
    y_label: str,
    fontsize: int = 18,
):
    plt.figure()
    plt.bar(x_axis, y_axis, color="#FF6600", yerr=yerr, capsize=6)
    plt.ylim(0.09, max(y_axis) * 1.3)
    plt.xlabel(xlabel=x_label, fontsize=fontsize)
    plt.xticks(np.arange(min(x_axis), max(x_axis) + 1, 1))
    plt.ylabel(ylabel=y_label, fontsize=fontsize, labelpad=10)
    plt.savefig(cfg.paths.results.bar_plot, bbox_inches="tight")
    plt.close()


def make_histogram_plot(
    cfg: Config,
    data: list,
    mean: float,
    x_label: str,
    bins: int = 5,
    file_tag: Union[str, None] = None,
    fontsize: int = 18,
) -> None:
    plt.figure()
    plt.hist(x=data, bins=bins, color="#0073FF", edgecolor="black", alpha=0.7)
    zero_shot_rmse = 0.12683835625648499  # For the SAA dataset
    # zero_shot_rmse = 0.5503723621368408  # For the OC20 dataset
    line_zero = plt.axvline(
        zero_shot_rmse,
        color="#6FFF00",
        linestyle="--",
        linewidth=2,
        label=f"Zero-shot RMSE = {zero_shot_rmse:.2f}",
    )
    line_mean = plt.axvline(
        mean,
        color="#FF6600",
        linestyle="--",
        linewidth=2,
        label=f"Mean {cfg.max_samples}-shot RMSE = {mean:.2f}",
    )
    improvement_fraction = np.sum(np.array(data) < zero_shot_rmse) / len(data)
    dummy = Line2D(
        [], [], color="none", label=f"Fraction improved: {improvement_fraction:.2f}"
    )

    # order: zero-shot first, then mean, then improvement at bottom
    handles = [line_zero, line_mean, dummy]
    labels = [h.get_label() for h in handles]
    plt.legend(handles, labels, loc="best")
    plt.xlabel(xlabel=x_label, fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize, labelpad=10)
    plt.savefig(
        cfg.paths.results.visualizations / f"histogram_{file_tag}.png",
        bbox_inches="tight",
    )
    plt.close()


def make_summary_figure(
    cfg, df, results, filename="n_shot_summary.png", fontsize=14, tick_fontsize=14
):
    """
    Combine all main plots (first parity, final parity, final histogram, bar summary)
    into one 2x2 figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # ----------------------
    # (1) Parity (first n)
    # ----------------------
    ax = axes[0, 0]

    # Build tidy dataframe for seaborn (Polars -> Pandas)
    df_plot = df.unpivot(index=[cfg.y_key]).with_columns(pl.col("variable")).to_pandas()

    # Choose a consistent color per expert
    experts = list(df_plot["variable"].unique())
    palette = sns.color_palette(n_colors=len(experts))
    color_map = {var: palette[i] for i, var in enumerate(experts)}

    # Plot points + per-expert regression; compute per-expert RMSE
    handles, labels = [], []
    for var in experts:
        sub = df_plot[df_plot["variable"] == var].copy()
        sub = sub.dropna(subset=["value", cfg.y_key])  # safety

        # Scatter
        sns.scatterplot(
            data=sub,
            x="value",
            y=cfg.y_key,
            ax=ax,
            color=color_map[var],
            alpha=0.6,
            s=25,
            legend=False,
        )

        # Regression line (no scatter)
        sns.regplot(
            data=sub,
            x="value",
            y=cfg.y_key,
            ax=ax,
            scatter=False,
            ci=None,
            color=color_map[var],
        )

        # RMSE for this expert
        err = sub["value"].to_numpy() - sub[cfg.y_key].to_numpy()
        rmse = float(np.sqrt(np.mean(err**2)))

        # Legend handle with matching color
        h = plt.Line2D([], [], color=color_map[var], linewidth=2)
        handles.append(h)
        labels.append(f"{var}: {rmse:.3f} eV")

    # Parity line (y = x) spanning all data
    min_val = min(df_plot["value"].min(), df_plot[cfg.y_key].min())
    max_val = max(df_plot["value"].max(), df_plot[cfg.y_key].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, zorder=2)

    # Legend and labels
    ax.legend(
        handles,
        labels,
        title="MLIP RMSE",
        fontsize=fontsize - 2,
        title_fontsize=fontsize - 1,
    )
    ax.set_xlabel("MLIP energy (eV)", fontsize=fontsize)
    ax.set_ylabel(f"{cfg.y_key} energy (eV)", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # ----------------------
    # (2) Parity (final n)
    # ----------------------
    ax = axes[0, 1]

    if (
        results.get("parity_first") is not None
        and results.get("parity_final") is not None
    ):
        p0 = results["parity_first"]
        pf = results["parity_final"]

        x0 = np.asarray(p0["x"]).ravel()
        y0 = np.asarray(p0["y"]).ravel()

        xf = np.asarray(pf["x"]).ravel()
        yf = np.asarray(pf["y"]).ravel()

        # parity line limits across BOTH sets
        xmin = float(np.nanmin(np.concatenate([x0, xf])))
        xmax = float(np.nanmax(np.concatenate([x0, xf])))

        # Scatter points (no legend entries)
        ax.scatter(x0, y0, alpha=0.35, s=15, marker="o")  # 0-shot points
        ax.scatter(xf, yf, alpha=0.35, s=15, marker="^")  # final-shot points

        # Parity line (y = x)
        ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=2, zorder=1)

        # 0-shot regression
        m0 = LinearRegression().fit(x0.reshape(-1, 1), y0)
        xfit = np.linspace(xmin, xmax, 200)
        yfit0 = m0.predict(xfit.reshape(-1, 1))
        (line0,) = ax.plot(xfit, yfit0, lw=2, label=results["parity_first"]["inset"])

        # final-shot regression
        mf = LinearRegression().fit(xf.reshape(-1, 1), yf)
        yfitf = mf.predict(xfit.reshape(-1, 1))
        (linef,) = ax.plot(xfit, yfitf, lw=2, label=results["parity_final"]["inset"])

        # Legend with ONLY the two fit lines
        ax.legend(
            handles=[line0, linef],
            fontsize=fontsize - 2,
            loc="upper left",
            frameon=False,
        )

        # Labels and title
        ax.set_xlabel("Ensemble energy (eV)", fontsize=fontsize)
        ax.set_ylabel("DFT energy (eV)", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    else:
        ax.set_visible(False)

    # ----------------------
    # (3) Histogram (final n)
    # ----------------------
    if "histogram_final" in results and results["histogram_final"] is not None:
        h = results["histogram_final"]
        ax = axes[1, 0]
        ax.hist(
            h["rmse_list"],
            bins=11,
            alpha=0.7,
            color="#0073FF",
            edgecolor="black",
        )
        # zero_shot_rmse = 0.5503723621368408  # For the OC20 dataset
        ax.axvline(
            results["parity_first"]["rmse"],
            color="#58CA00",
            linestyle="--",
            linewidth=2,
            label=f"Zero-shot RMSE = {results['parity_first']['rmse']:.3f} eV",
        )
        # Compute and add improvement fraction
        improvement_fraction = np.sum(
            np.array(h["rmse_list"], dtype=float).ravel()
            < results["parity_first"]["rmse"]
        ) / len(h["rmse_list"])
        legend_text = f"Fraction improved: {improvement_fraction:.2f}"

        # Legend with extra line of text
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="none"))
        labels.append(legend_text)

        ax.legend(
            handles,
            labels,
            fontsize=fontsize - 2,
            loc="upper right",
            frameon=True,
            facecolor="white",
        )
        ax.set_xlabel(f"{cfg.max_samples}-shot RMSE (eV)", fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    else:
        axes[1, 0].set_visible(False)

    # ----------------------
    # (4) Bar plot summary
    # ----------------------
    if "summary" in results and "bar" in results["summary"]:
        b = results["summary"]["bar"]
        ax = axes[1, 1]
        ax.bar(b["x"], b["y"], color="#FF6600", yerr=b["yerr"], alpha=0.8, capsize=4)
        ax.set_ylim(0.09, max(b["y"]) * 1.3)
        ax.set_xlabel(b["x_label"], fontsize=fontsize)
        ax.set_ylabel(b["y_label"], fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    else:
        axes[1, 1].set_visible(False)

    # Save and show
    plt.tight_layout()
    # Label each subplot
    panel_labels = ["a)", "b)", "c)", "d)"]
    x_offset = -0.04
    y_offset = 0.05
    positions = [
        (0.05 + x_offset, 0.95 + y_offset),
        (0.55 + x_offset, 0.95 + y_offset),
        (0.05 + x_offset, 0.47 + y_offset),
        (0.55 + x_offset, 0.47 + y_offset),
    ]

    for label, (x, y) in zip(panel_labels, positions):
        fig.text(
            x, y, label, fontsize=fontsize + 2, fontweight="bold", va="top", ha="left"
        )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved combined figure to {filename}")

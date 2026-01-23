from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
        title="MLIP $\\mathrm{{RMSE}}_{{parity}}$",
        fontsize=fontsize - 2,
        title_fontsize=fontsize - 1,
    )
    ax.set_xlabel("MLIP $\\mathrm{E}_{ads}$ (eV)", fontsize=fontsize)
    ax.set_ylabel(f"{cfg.y_key} $\\mathrm{{E}}_{{ads}}$ (eV)", fontsize=fontsize)
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
        rmse0 = results["parity_first"]["rmse"]
        rmse0_bf = results["parity_first"].get("rmse_best_fit")
        label0 = (
            f"0-shot $\\mathrm{{RMSE}}_{{parity}}$ = {rmse0:.3f}   "
            f"$\\mathrm{{RMSE}}_{{fit}}$ = {rmse0_bf:.3f}"
            if rmse0_bf is not None
            else f"0-shot $\\mathrm{{RMSE}}_{{parity}}$ = {rmse0:.3f}"
        )
        (line0,) = ax.plot(xfit, yfit0, lw=2, label=label0)

        # final-shot regression
        mf = LinearRegression().fit(xf.reshape(-1, 1), yf)
        yfitf = mf.predict(xfit.reshape(-1, 1))
        rmsef = results["parity_final"]["rmse"]
        rmsef_bf = results["parity_final"].get("rmse_best_fit")
        labelf = (
            f"{cfg.max_samples}-shot $\\mathrm{{RMSE}}_{{parity}}$ = {rmsef:.3f}   "
            f"$\\mathrm{{RMSE}}_{{fit}}$ = {rmsef_bf:.3f}"
            if rmsef_bf is not None
            else f"{cfg.max_samples}-shot $\\mathrm{{RMSE}}_{{parity}}$ = {rmsef:.3f}"
        )
        (linef,) = ax.plot(xfit, yfitf, lw=2, label=labelf)

        # Legend with ONLY the two fit lines
        ax.legend(
            handles=[line0, linef],
            fontsize=fontsize - 3,
            loc="upper left",
            frameon=False,
        )

        # Labels and title
        ax.set_xlabel("Ensemble $\\mathrm{E}_{ads}$ (eV)", fontsize=fontsize)
        ax.set_ylabel("DFT $\\mathrm{E}_{ads}$ (eV)", fontsize=fontsize)
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
            label=f"Zero-shot $\\mathrm{{RMSE}}_{{parity}}$ = {results['parity_first']['rmse']:.3f} eV",
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
        ax.set_ylim(0.0, max(b["y"]) * 1.3)
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


def get_miscalibration_area(
    confidence_levels: np.ndarray, coverages: np.ndarray
) -> float:
    """
    Compute the area between the calibration curve and the ideal y=x line.
    """
    confidence_levels = np.asarray(confidence_levels, dtype=float)
    coverages = np.asarray(coverages, dtype=float)
    return float(np.trapz(np.abs(coverages - confidence_levels), confidence_levels))


def plot_calibration_curve(
    confidence_levels: np.ndarray,
    coverages: np.ndarray,
    coverage_stds: np.ndarray,
    fontsize: int = 12,
    tick_fontsize: Union[int, None] = None,
    ax: plt.Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ideal = np.linspace(0, 1, 100)
    line1 = ax.errorbar(
        confidence_levels,
        coverages,
        yerr=coverage_stds,
        fmt="o-",
        label="Model",
        capsize=4,
    )
    (line2,) = ax.plot(ideal, ideal, linestyle="--", color="black", label="Ideal")

    legend1 = ax.legend(
        handles=[line1, line2], loc="upper left", fontsize=fontsize * 0.8
    )
    ax.add_artist(legend1)

    miscalibration_area = get_miscalibration_area(confidence_levels, coverages)
    phantom_patch = Patch(color="none", label=f"MisCal = {miscalibration_area:.3f}")
    ax.legend(handles=[phantom_patch], loc="lower right", fontsize=fontsize * 0.7)

    ax.set_xlabel("Expected Confidence Level", fontsize=fontsize)
    ax.set_ylabel("Observed Confidence Level", fontsize=fontsize)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize if tick_fontsize is None else tick_fontsize,
    )


def plot_sharpness(
    y_pred_std: np.ndarray,
    fontsize: int = 12,
    tick_fontsize: Union[int, None] = None,
    ax: plt.Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(y_pred_std, bins=20, kde=False, ax=ax)
    ax.set_xlabel("Standard Deviation (eV)", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)

    sharpness = float(np.mean(y_pred_std))
    dispersion = (
        float(np.std(y_pred_std) / sharpness) if sharpness != 0 else float("nan")
    )

    ax.axvline(
        sharpness,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Sha = {sharpness:.3f} eV",
    )
    ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
    ax.plot([], [], " ", label=rf"$C_{{V}}$ = {dispersion:.3f}")
    ax.legend(fontsize=fontsize * 0.6)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize if tick_fontsize is None else tick_fontsize,
    )


def make_uncertainty_summary_figure(
    cfg: Config,
    y: np.ndarray,
    y_preds: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: np.ndarray,
    coverages: np.ndarray,
    coverage_stds: np.ndarray,
    sharpness_summary: Union[dict, None] = None,
    filename: Union[str, Path, None] = None,
    fontsize: int = 12,
    tick_fontsize: int = 14,
    subset_size: Union[int, None] = None,
) -> None:
    """
    Combine parity, calibration curve, sharpness, and optional sharpness-by-n panel.
    """
    if filename is None:
        filename = cfg.paths.results.visualizations / "uncertainty_summary.png"

    use_sharpness_panel = sharpness_summary is not None
    if use_sharpness_panel:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        plt.subplots_adjust(wspace=0.3, hspace=0.35)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # ----------------------
    # (1) Parity with error bars
    # ----------------------
    ax = axes[0, 0] if use_sharpness_panel else axes[0]
    x_grid = np.linspace(float(np.min(y_preds)), float(np.max(y_preds)), 100)
    ax.plot(x_grid, x_grid, color="black", linewidth=2, linestyle="--")
    model = LinearRegression().fit(y_preds.reshape(-1, 1), y.reshape(-1, 1))
    ax.plot(x_grid, model.predict(x_grid.reshape(-1, 1)), color="blue", linewidth=2)

    n_subset = subset_size if subset_size is not None else len(y_preds)
    n_subset = min(n_subset, len(y_preds))
    idx = np.random.choice(len(y_preds), size=n_subset, replace=False)
    y_subset = y[idx]
    y_preds_subset = y_preds[idx]
    y_pred_std_subset = y_pred_std[idx]

    ax.errorbar(
        y_preds_subset,
        y_subset,
        yerr=y_pred_std_subset,
        color="blue",
        fmt="o",
        capsize=5,
        alpha=0.3,
        markeredgecolor="none",
    )
    ax.set_xlabel("Ensemble $\\mathrm{E}_{ads}$ (eV)", fontsize=fontsize)
    ax.set_ylabel("DFT $\\mathrm{E}_{ads}$ (eV)", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # ----------------------
    # (2) Calibration curve
    # ----------------------
    ax = axes[0, 1] if use_sharpness_panel else axes[1]
    plot_calibration_curve(
        confidence_levels=confidence_levels,
        coverages=coverages,
        coverage_stds=coverage_stds,
        fontsize=fontsize,
        tick_fontsize=tick_fontsize,
        ax=ax,
    )

    # ----------------------
    # (3) Sharpness histogram
    # ----------------------
    ax = axes[1, 0] if use_sharpness_panel else axes[2]
    plot_sharpness(
        y_pred_std=y_pred_std, fontsize=fontsize, tick_fontsize=tick_fontsize, ax=ax
    )

    # ----------------------
    # (4) Sharpness vs holdout samples
    # ----------------------
    if use_sharpness_panel:
        ax = axes[1, 1]
        n_values = np.asarray(sharpness_summary["n_values"])
        sharpness_mean = np.asarray(sharpness_summary["sharpness_mean"])
        sharpness_std = np.asarray(sharpness_summary["sharpness_std"])
        ax.bar(
            n_values,
            sharpness_mean,
            yerr=sharpness_std,
            color=(81 / 255, 140 / 255, 180 / 255, 1.0),
            edgecolor="black",
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.5},
        )
        ax.set_xlabel("Number of holdouts", fontsize=fontsize)
        ax.set_ylabel("Sharpness (eV)", fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    plt.tight_layout()

    label_dx = -0.01
    label_dy = 0.03
    if use_sharpness_panel:
        panel_labels = ["a)", "b)", "c)", "d)"]
        positions = [
            (0.05 + label_dx, 0.96 + label_dy),
            (0.52 + label_dx, 0.96 + label_dy),
            (0.05 + label_dx, 0.48 + label_dy),
            (0.52 + label_dx, 0.48 + label_dy),
        ]
    else:
        panel_labels = ["a)", "b)", "c)"]
        positions = [
            (0.05 - 0.045 + label_dx, 0.95 + 0.05 + label_dy),
            (0.40 - 0.045 + label_dx, 0.95 + 0.05 + label_dy),
            (0.75 - 0.045 + label_dx, 0.95 + 0.05 + label_dy),
        ]

    for label, (x, y_pos) in zip(panel_labels, positions):
        fig.text(
            x,
            y_pos,
            label,
            fontsize=fontsize + 2,
            fontweight="bold",
            va="top",
            ha="left",
        )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined figure to {filename}")

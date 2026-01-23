import random
from matplotlib import pyplot as plt
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Union, Dict, Any

from trot.config import Config
from trot.coverage import get_fsc_metric
from trot.evaluate import get_rmse
from trot.processing import df_to_numpy, get_holdout_split
from trot.visualize import (
    make_bar_plot,
    make_parity_plot,
    make_histogram_plot,
    make_uncertainty_summary_figure,
)


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """
    Approximate inverse CDF for a standard normal distribution.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p, dtype=float)
    mask_low = p < plow
    mask_high = p > phigh
    mask_mid = ~(mask_low | mask_high)

    if np.any(mask_low):
        q = np.sqrt(-2 * np.log(p[mask_low]))
        x[mask_low] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    if np.any(mask_mid):
        q = p[mask_mid] - 0.5
        r = q * q
        x[mask_mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )
        )

    if np.any(mask_high):
        q = np.sqrt(-2 * np.log(1 - p[mask_high]))
        x[mask_high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    return x


def z_score_from_confidence(confidence_level: np.ndarray) -> np.ndarray:
    """
    Return z-score for a two-sided normal interval with given confidence.
    """
    p = 0.5 + np.asarray(confidence_level, dtype=float) / 2.0
    return _normal_ppf(p)


def get_calibration_data(
    residuals: np.ndarray,
    y_pred_std: np.ndarray,
    calibration_factor: float,
    confidence_levels: np.ndarray = np.linspace(0.05, 0.95, 19),
) -> tuple[np.ndarray, np.ndarray]:
    z_scores = z_score_from_confidence(confidence_levels)
    thresholds = z_scores[:, None] * y_pred_std[None, :] * calibration_factor
    residuals_abs = np.abs(residuals)[None, :]
    within_interval = residuals_abs <= thresholds
    coverages = within_interval.mean(axis=1)
    return confidence_levels, coverages


def get_avg_std(
    df: pl.DataFrame,
    avg_alias: str,
    std_alias: str,
) -> pl.DataFrame:
    average = df.mean_horizontal().alias(avg_alias)
    std = pl.concat_list(df).list.std().alias(std_alias)
    df_avg_std = df.select(average, std)
    return df_avg_std


def modify_data(
    X: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    linearize: bool,
) -> np.ndarray:
    if linearize:
        Xh = np.asarray(X_holdout)
        yh = np.asarray(y_holdout).reshape(-1, 1)

        if Xh.ndim == 1:
            mu_h = Xh.reshape(-1, 1)  # (n,1)
        else:
            mu_h = Xh.mean(axis=1, keepdims=True)  # (n,1)

        lr = LinearRegression().fit(mu_h, yh)
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        X = a * X + b
        return X
    else:
        # mean_residual = y_holdout.mean() - X_holdout.mean()  # Ensemble bias
        # X += mean_residual
        residuals = y_holdout[:, None] - X_holdout  # Model biases
        mean_residuals = residuals.mean(axis=0)
        X += mean_residuals
        return X


def remove_outliers(X: np.ndarray, std_factor: float = 1.0) -> np.ndarray:
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    mask = np.abs(X - mean) <= std * std_factor
    X_filtered = np.where(mask, X, np.nan)
    return X_filtered


def n_shot(
    cfg: Config,
    df: pl.DataFrame,
    df_holdout: Union[pl.DataFrame, None] = None,
    max_samples: int = 2,
    fsc_bins: int = 6,
    hist_bins: int = 6,
    linearize: bool = False,
    plot_now: bool = True,
    seed: Union[int, None] = 0,
) -> Dict[str, Any]:
    """
    Runs the n-shot evaluation and returns a results dictionary containing:
      • RMSE data for each n
      • Summary statistics across n
      • Only the final parity snapshot (largest n)
      • Only the final histogram data (largest n)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results: Dict[str, Any] = {
        "settings": {
            "max_samples": max_samples,
            "fsc_bins": fsc_bins,
            "hist_bins": hist_bins,
            "linearize": linearize,
            "removal_iterations": cfg.removal_iterations,
            "std_factor": cfg.std_factor,
            "y_key": cfg.y_key,
        },
        "per_n": {},  # holds RMSE stats per n
        "summary": {},  # aggregate across n
        "parity_final": None,
        "histogram_final": None,
    }

    if linearize:
        # 0-shot plus 2..max_samples (inclusive), skipping n=1
        sample_range = [0] + list(range(2, max_samples + 1))
    else:
        sample_range = range(0, max_samples + 1)
    rmse_mean_list = []
    rmse_std_list = []
    rmse_best_fit_mean_list = []
    rmse_best_fit_std_list = []
    summary_ns = []

    num_samples = df_holdout.height if df_holdout is not None else df.height

    for n in sample_range:
        rmse_list = []
        rmse_best_fit_list = []

        # Generate random holdout index combinations
        max_combos = 1000
        if n == 0:
            combos = [tuple()]  # single empty holdout
        else:
            all_indices = list(range(num_samples))
            combos = [
                tuple(sorted(random.sample(all_indices, n))) for _ in range(max_combos)
            ]

        for holdout_indices in combos:
            X, y = df_to_numpy(df, cfg.y_key)

            # Split into train/holdout sets
            if df_holdout is not None:
                X_source, y_source = df_to_numpy(df_holdout, cfg.y_key)
                _, _, X_holdout, y_holdout = get_holdout_split(
                    X=X_source, y=y_source, holdout_indices=holdout_indices
                )
            else:
                X, y, X_holdout, y_holdout = get_holdout_split(
                    X=X, y=y, holdout_indices=holdout_indices
                )

            # Modify data (few-shot)
            if n >= 1:
                X = modify_data(
                    X=X,
                    X_holdout=X_holdout,
                    y_holdout=y_holdout,
                    linearize=linearize,
                )

            # Remove outliers
            for _ in range(cfg.removal_iterations):
                X = remove_outliers(X, std_factor=cfg.std_factor)

            # Compute stats
            mean = np.nanmean(X, axis=1).reshape(-1, 1)
            std = np.nanstd(X, axis=1)
            pred_intervals = std
            rmse = get_rmse(y_pred=mean, y=y)
            best_fit = LinearRegression().fit(mean, y)
            best_fit_pred = best_fit.predict(mean)
            best_fit_rmse = get_rmse(y_pred=best_fit_pred, y=y)
            rmse_list.append(float(rmse))
            rmse_best_fit_list.append(float(best_fit_rmse))

            # If this is the first n, keep parity snapshot
            if n == 0:
                results["parity_first"] = {
                    "x": mean,
                    "y": y.copy(),
                    "yerr": pred_intervals.copy(),
                    "rmse": rmse,
                    "rmse_best_fit": best_fit_rmse,
                    "inset": f"{n}-shot RMSE: {rmse:.3f} eV",
                }

            # If this is the final n, keep last parity snapshot
            if n == sample_range[-1]:
                results["parity_final"] = {
                    "x": mean,
                    "y": y.copy(),
                    "yerr": pred_intervals.copy(),
                    "inset": f"{n}-shot RMSE: {rmse:.3f} eV",
                    "rmse": rmse,
                    "rmse_best_fit": best_fit_rmse,
                }

        # Compute summary stats
        rmse_mean = float(np.mean(rmse_list)) if rmse_list else float("nan")
        rmse_std = float(np.std(rmse_list)) if rmse_list else float("nan")
        rmse_best_fit_mean = (
            float(np.mean(rmse_best_fit_list)) if rmse_best_fit_list else float("nan")
        )
        rmse_best_fit_std = (
            float(np.std(rmse_best_fit_list)) if rmse_best_fit_list else float("nan")
        )

        rmse_mean_list.append(rmse_mean)
        rmse_std_list.append(rmse_std)
        rmse_best_fit_mean_list.append(rmse_best_fit_mean)
        rmse_best_fit_std_list.append(rmse_best_fit_std)
        summary_ns.append(n)

        # Save per-n stats (lightweight)
        results["per_n"][n] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_best_fit_mean": rmse_best_fit_mean,
            "rmse_best_fit_std": rmse_best_fit_std,
        }

        # If this is the final n, store histogram data
        if n == sample_range[-1]:
            results["histogram_final"] = {
                "rmse_list": rmse_list,
                "rmse_mean": rmse_mean,
                "hist_bins": hist_bins,
                "rmse_best_fit_list": rmse_best_fit_list,
                "rmse_best_fit_mean": rmse_best_fit_mean,
            }

        # Optional plotting for current n
        if plot_now and n == sample_range[-1]:
            make_histogram_plot(
                cfg=cfg,
                data=rmse_list,
                mean=rmse_mean,
                x_label="RMSE (eV)",
                bins=hist_bins,
                file_tag=f"holdouts_{n}",
            )

    # Summary (for bar plot)
    results["summary"] = {
        "n_values": summary_ns,
        "bar": {
            "x": summary_ns,
            "y": rmse_mean_list,
            "yerr": rmse_std_list,
            "x_label": "Number of holdouts",
            "y_label": "Mean $\\mathrm{{RMSE}}_{{parity}}$ (eV)",
        },
        "best_fit_bar": {
            "x": summary_ns,
            "y": rmse_best_fit_mean_list,
            "yerr": rmse_best_fit_std_list,
            "x_label": "Number of holdouts",
            "y_label": "Mean Best-Fit RMSE (eV)",
        },
    }

    # Optional final bar + parity plots
    if plot_now:
        make_bar_plot(
            cfg=cfg,
            x_axis=results["summary"]["bar"]["x"],
            y_axis=results["summary"]["bar"]["y"],
            yerr=results["summary"]["bar"]["yerr"],
            x_label=results["summary"]["bar"]["x_label"],
            y_label=results["summary"]["bar"]["y_label"],
        )

        if results["parity_final"] is not None:
            p = results["parity_final"]
            make_parity_plot(
                cfg=cfg,
                x_axis=p["x"],
                y_axis=p["y"],
                yerr=p["yerr"],
                x_label="Ensemble energy (eV)",
                y_label="DFT energy (eV)",
                inset=p["inset"],
            )

    return results


def uncertainty_analysis(
    cfg: Config,
    df: pl.DataFrame,
    n: int,
    df_holdout: Union[pl.DataFrame, None] = None,
    max_combos: int = 1000,
    calibration_factor: float = 1.0,
    confidence_levels: np.ndarray = np.linspace(0.05, 0.95, 19),
    linearize: bool = False,
    plot_now: bool = True,
    seed: Union[int, None] = 0,
    summary_filename: Union[str, None] = None,
    fontsize: int = 12,
    tick_fontsize: int = 14,
    subset_size: Union[int, None] = None,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    X_full, y_full = df_to_numpy(df=df, y_col=cfg.y_key)
    num_samples = df_holdout.height if df_holdout is not None else df.height

    if n == 0:
        combos = [tuple()]
    else:
        all_indices = list(range(num_samples))
        combos = [tuple(sorted(random.sample(all_indices, n))) for _ in range(max_combos)]

    coverage_lists = []
    parity_snapshot = None

    for holdout_indices in combos:
        X, y = X_full.copy(), y_full.copy()
        if df_holdout is not None:
            X_source, y_source = df_to_numpy(df=df_holdout, y_col=cfg.y_key)
            _, _, X_holdout, y_holdout = get_holdout_split(
                X=X_source, y=y_source, holdout_indices=holdout_indices
            )
        else:
            X, y, X_holdout, y_holdout = get_holdout_split(
                X=X, y=y, holdout_indices=holdout_indices
            )

        if n >= 1:
            X = modify_data(
                X=X,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
                linearize=linearize,
            )

        for _ in range(cfg.removal_iterations):
            X = remove_outliers(X, std_factor=cfg.std_factor)

        y_preds = np.nanmean(X, axis=1)
        y_pred_std = np.nanstd(X, axis=1)
        residuals = y - y_preds
        _, coverages = get_calibration_data(
            y_pred_std=y_pred_std,
            residuals=residuals,
            calibration_factor=calibration_factor,
            confidence_levels=confidence_levels,
        )
        coverage_lists.append(coverages)

        if parity_snapshot is None:
            parity_snapshot = {
                "y": y,
                "y_preds": y_preds,
                "y_pred_std": y_pred_std,
            }

    average_coverages = np.mean(coverage_lists, axis=0)
    coverage_stds = np.std(coverage_lists, axis=0)

    results = {
        "settings": {
            "n": n,
            "max_combos": max_combos,
            "calibration_factor": calibration_factor,
            "linearize": linearize,
            "removal_iterations": cfg.removal_iterations,
            "std_factor": cfg.std_factor,
            "y_key": cfg.y_key,
        },
        "parity": parity_snapshot,
        "calibration": {
            "confidence_levels": confidence_levels,
            "coverages": average_coverages,
            "coverage_stds": coverage_stds,
        },
    }

    if plot_now and parity_snapshot is not None:
        make_uncertainty_summary_figure(
            cfg=cfg,
            y=parity_snapshot["y"],
            y_preds=parity_snapshot["y_preds"],
            y_pred_std=parity_snapshot["y_pred_std"],
            confidence_levels=confidence_levels,
            coverages=average_coverages,
            coverage_stds=coverage_stds,
            filename=summary_filename,
            fontsize=fontsize,
            tick_fontsize=tick_fontsize,
            subset_size=subset_size,
        )

    return results


def greedy_cost(
    mean: np.ndarray,
    std: np.ndarray,
    target: float,
    alpha: float = 0.75,  # Equal weighting of accuracy and uncertainty
) -> np.ndarray:
    range_mu = np.ptp(mean)  # max - min
    range_sigma = np.ptp(std)  # max - min
    # Add a small epsilon to avoid division by zero if range is zero
    epsilon = 1e-8
    range_mu = range_mu if range_mu > 0 else epsilon
    range_sigma = range_sigma if range_sigma > 0 else epsilon
    accuracy_term = np.abs(mean - target) / range_mu
    uncertainty_term = std / range_sigma
    cost = alpha * accuracy_term + (1 - alpha) * uncertainty_term
    return cost


def plot_candidates(
    mean: np.ndarray,
    std: np.ndarray,
    cost: np.ndarray,
    target: float,
    top_n: int = 10,
) -> None:
    sorted_indices = np.argsort(cost)
    top_indices = sorted_indices[:top_n]
    plt.figure(figsize=(8, 5))
    for rank, i in enumerate(top_indices):
        plt.errorbar(
            x=rank + 1, y=mean[i], yerr=std[i], fmt="o", label=f"Candidate {i}"
        )
    plt.axhline(y=target, color="r", linestyle="--", label="Target")
    plt.xlabel("Candidate Rank (by cost)")
    plt.ylabel("Binding Energy (eV)")
    plt.title("Top Candidates by Cost")
    # plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def get_recommendation(
    cfg: Config,
    df: pl.DataFrame,
    cost_fn: callable,
    df_holdout: Union[pl.DataFrame, None] = None,
) -> pl.DataFrame:
    X, _ = df_to_numpy(df, cfg.y_key)
    X_holdout, y_holdout = df_to_numpy(df_holdout, cfg.y_key)
    X = modify_data(
        X=X, X_holdout=X_holdout, y_holdout=y_holdout, linearize=cfg.linearize
    )
    mean = np.nanmean(X, axis=1)
    std = np.nanstd(X, axis=1)
    cost = cost_fn(mean, std, target=cfg.target, alpha=0.75)
    plot_candidates(mean, std, cost, target=cfg.target)
    minimum_index = np.argmin(cost)
    return minimum_index

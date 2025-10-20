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
)


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
    y: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    linearize: bool,
) -> tuple[np.ndarray, np.ndarray]:
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
        return X, y
    else:
        mean_residual = y_holdout.mean() - X_holdout.mean()
        print(f"Mean residual to add: {mean_residual:.3f} eV")
        X += mean_residual
        return X, y


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
    summary_ns = []

    num_samples = df_holdout.height if df_holdout is not None else df.height

    for n in sample_range:
        rmse_list = []

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
                X, y = modify_data(
                    X=X,
                    y=y,
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
            rmse_list.append(float(rmse))

            # If this is the first n, keep parity snapshot
            if n == 0:
                results["parity_first"] = {
                    "x": mean,
                    "y": y.copy(),
                    "yerr": pred_intervals.copy(),
                    "rmse": rmse,
                    "inset": f"{n}-shot RMSE: {rmse:.3f} eV",
                }

            # If this is the final n, keep last parity snapshot
            if n == sample_range[-1]:
                results["parity_final"] = {
                    "x": mean,
                    "y": y.copy(),
                    "yerr": pred_intervals.copy(),
                    "inset": f"{n}-shot RMSE: {rmse:.3f} eV",
                }

        # Compute summary stats
        rmse_mean = float(np.mean(rmse_list)) if rmse_list else float("nan")
        rmse_std = float(np.std(rmse_list)) if rmse_list else float("nan")

        rmse_mean_list.append(rmse_mean)
        rmse_std_list.append(rmse_std)
        summary_ns.append(n)

        # Save per-n stats (lightweight)
        results["per_n"][n] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
        }

        # If this is the final n, store histogram data
        if n == sample_range[-1]:
            results["histogram_final"] = {
                "rmse_list": rmse_list,
                "rmse_mean": rmse_mean,
                "hist_bins": hist_bins,
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
            "y_label": "Mean RMSE (eV)",
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
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def get_recommendation(
    cfg: Config,
    df: pl.DataFrame,
    cost_fn: callable,
    df_holdout: Union[pl.DataFrame, None] = None,
) -> pl.DataFrame:
    X, y = df_to_numpy(df, cfg.y_key)
    X_holdout, y_holdout = df_to_numpy(df_holdout, cfg.y_key)
    X, y = modify_data(
        X=X, y=y, X_holdout=X_holdout, y_holdout=y_holdout, linearize=cfg.linearize
    )
    mean = np.nanmean(X, axis=1)
    std = np.nanstd(X, axis=1)
    cost = cost_fn(mean, std, target=cfg.target, alpha=0.75)
    plot_candidates(mean, std, cost, target=cfg.target)
    minimum_index = np.argmin(cost)
    return minimum_index

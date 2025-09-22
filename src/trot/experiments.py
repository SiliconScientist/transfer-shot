import random
from matplotlib import pyplot as plt
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Union

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
        model = LinearRegression().fit(X_holdout, y_holdout)
        X = model.coef_ * X + model.intercept_
        return X, y
    else:
        X += y_holdout.mean() - X_holdout.mean()
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
) -> None:
    sample_range = range(2 if linearize else 0, max_samples + 1)
    rmse_mean_list = []
    rmse_std_list = []
    for n in sample_range:
        rmse_list = []
        if df_holdout is not None:
            num_samples = df_holdout.height
        else:
            num_samples = df.height
        max_combos = 1000
        all_indices = list(range(num_samples))
        combos = [
            tuple(sorted(random.sample(all_indices, n))) for _ in range(max_combos)
        ]
        for holdout_indices in combos:
            X, y = df_to_numpy(df, cfg.y_key)
            # If you're using a given dataset as the holdout set, then you'll
            # make those the holdout samples
            if df_holdout is not None:
                X_source, y_source = df_to_numpy(df_holdout, cfg.y_key)
                _, _, X_holdout, y_holdout = get_holdout_split(
                    X=X_source,
                    y=y_source,
                    holdout_indices=holdout_indices,
                )
            # Otherwise, you'll split your original dataset to make holdout set
            else:
                X, y, X_holdout, y_holdout = get_holdout_split(
                    X=X,
                    y=y,
                    holdout_indices=holdout_indices,
                )
            if n >= 1:
                X, y = modify_data(
                    X=X,
                    y=y,
                    X_holdout=X_holdout,
                    y_holdout=y_holdout,
                    linearize=linearize,
                )
            for _ in range(cfg.removal_iterations):
                X = remove_outliers(X, std_factor=cfg.std_factor)
            mean = np.nanmean(X, axis=1).reshape(-1, 1)
            std = np.nanstd(X, axis=1)
            pred_intervals = 0.75 * std
            rmse = get_rmse(y_pred=mean, y=y)
            rmse_list.append(rmse)
            fsc = get_fsc_metric(
                X=mean.squeeze(), std=pred_intervals, y=y, bins=fsc_bins
            )
        make_parity_plot(
            cfg=cfg,
            x_axis=mean,
            y_axis=y,
            yerr=pred_intervals,
            x_label="Average energy (eV)",
            y_label="DFT energy (eV)",
            title=f"Parity Plot with {holdout_indices} holdouts",
            inset=f"RMSE: {rmse:.3f} \n FSC (bins={fsc_bins}): {fsc:.2f} \n Removals: {cfg.removal_iterations}",
        )
        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        rmse_mean_list.append(rmse_mean)
        rmse_std_list.append(rmse_std)
        make_histogram_plot(
            cfg=cfg,
            data=rmse_list,
            mean=rmse_mean,
            x_label="RMSE (eV)",
            bins=hist_bins,
            file_tag=f"holdouts_{n}",
        )
    make_bar_plot(
        cfg=cfg,
        x_axis=sample_range,
        y_axis=rmse_mean_list,
        yerr=rmse_std_list,
        x_label="Number of holdouts",
        y_label="Mean RMSE (eV)",
        title="Mean RMSE vs Number of Holdouts",
    )


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

import itertools
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression

from trot.config import Config
from trot.coverage import get_fsc_metric
from trot.evaluate import get_rmse
from trot.processing import get_holdout_split
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
        X_mean = X_holdout.mean()
        y_mean = y_holdout.mean()
        difference = y_mean - X_mean
        X = X + difference
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
    max_samples: int = 2,
    fsc_bins: int = 1,
    hist_bins: int = 6,
    linearize: bool = False,
) -> None:
    sample_range = range(2 if linearize else 0, max_samples + 1)
    rmse_mean_list = []
    for n in sample_range:
        rmse_list = []
        for holdout_indices in itertools.combinations(range(1, df.height), n):
            X, y, X_holdout, y_holdout = get_holdout_split(
                df=df,
                y_col=cfg.y_key,
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
            rmse = get_rmse(y_pred=mean, y=y)
            rmse_list.append(rmse)
            fsc = get_fsc_metric(X=mean.squeeze(), std=std, y=y, bins=fsc_bins)
        make_parity_plot(
            cfg=cfg,
            x_axis=mean,
            y_axis=y,
            yerr=std.squeeze(),
            x_label="Average energy (eV)",
            y_label="DFT energy (eV)",
            title=f"Parity Plot with {holdout_indices} holdouts",
            inset=f"RMSE: {rmse:.3f} \n FSC (bins={fsc_bins}): {fsc:.2f} \n Removals: {cfg.removal_iterations}",
        )
        rmse_mean = np.mean(rmse_list)
        rmse_mean_list.append(rmse_mean)
        make_histogram_plot(
            cfg=cfg,
            data=rmse_list,
            mean=rmse_mean,
            x_label="RMSE (eV)",
            title=f"Histogram: Bins = {hist_bins}; Holdouts = {n}",
            bins=hist_bins,
            file_tag=f"holdouts_{n}",
        )
    make_bar_plot(
        cfg=cfg,
        x_axis=sample_range,
        y_axis=rmse_mean_list,
        x_label="Number of holdouts",
        y_label="Mean RMSE (eV)",
        title="Mean RMSE vs Number of Holdouts",
    )


# def iterative_averages(
#     cfg: Config,
#     df: pl.DataFrame,
#     y_col: str,
#     iterations: int = 4,
#     std_factor: int = 1,
#     avg_alias: str = "average",
#     std_alias: str = "std",
# ) -> None:
#     rmse_list = []
#     df_avgs = pl.DataFrame()
#     df_predictions = df.select(pl.exclude(y_col))
#     for i in range(iterations):
#         df_avg_std = get_avg_std(df_predictions, avg_alias, std_alias)
#         df_predictions = remove_outliers(
#             df=df_predictions,
#             df_avg_std=df_avg_std,
#             std_factor=std_factor,
#             avg_alias=avg_alias,
#             std_alias=std_alias,
#         )
#         df_y_avg_std = df_avg_std.with_columns(df.select(y_col))
#         X, _, y = get_dataframe_output(
#             df_y_avg_std=df_y_avg_std,
#             avg_alias=avg_alias,
#             std_alias=std_alias,
#             y_col=y_col,
#         )
#         rmse = get_rmse(y=y, y_pred=X)
#         rmse_list.append(rmse)
#         df_avg = df_avg_std.drop(std_alias).rename({avg_alias: f"{avg_alias}_{i}"})
#         df_avgs = df_avgs.hstack(df_avg)
#     make_bar_plot(
#         cfg=cfg,
#         x_axis=range(iterations),
#         y_axis=rmse_list,
#         x_label="Iterations",
#         y_label="RMSE (eV)",
#         title=f"RMSE vs Iterations with std_factor = {std_factor}",
#     )
#     df_y_avg = df_avgs.hstack(df.select(y_col))
#     make_multiclass_parity_plot(cfg=cfg, df=df_y_avg, y_col=y_col)
#     X, std, y = get_dataframe_output(
#         df_y_avg_std=df_y_avg_std,
#         avg_alias=avg_alias,
#         std_alias=std_alias,
#         y_col=y_col,
#     )
#     lower = X - std
#     upper = X + std
#     bins = 6
#     binned_indices = get_binned_indices(data=X.squeeze(), bins=bins)
#     fsc = get_fsc_metric(binned_indices, y, lower, upper)
#     rmse = get_rmse(y=y, y_pred=X)
#     inset = f"RMSE: {rmse:.3f} \n FSC (bins={bins}): {fsc:.3f}"
#     make_parity_plot(
#         cfg=cfg,
#         x_axis=X,
#         y_axis=y,
#         yerr=std.squeeze(),
#         x_label=f"{avg_alias}_{iterations - 1}",
#         y_label=y_col,
#         title=f"Parity Plot with std_factor = {std_factor}",
#         inset=inset,
#     )

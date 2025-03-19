import itertools
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import torch

from trot.config import Config
from trot.coverage import get_binned_indices, get_fsc_metric
from trot.evaluate import get_rmse
from trot.visualize import (
    make_bar_plot,
    make_parity_plot,
    make_histogram_plot,
    make_multiclass_parity_plot,
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


def remove_outliers(
    df: pl.DataFrame,
    df_avg_std: pl.DataFrame,
    avg_alias: str,
    std_alias: str,
    std_factor: int = 1,
) -> pl.DataFrame:
    df_filtered = df.with_columns(
        [
            pl.when(
                (df[col] - df_avg_std[avg_alias]).abs()
                <= df_avg_std[std_alias] * std_factor
            )
            .then(df[col])
            .otherwise(None)
            .alias(col)
            for col in df.columns
        ]
    )
    return df_filtered


def iterative_averages(
    cfg: Config,
    df: pl.DataFrame,
    y_col: str,
    iterations: int = 4,
    std_factor: int = 1,
    avg_alias: str = "average",
    std_alias: str = "std",
) -> None:
    rmse_list = []
    df_avgs = pl.DataFrame()
    df_predictions = df.select(pl.exclude(y_col))
    for i in range(iterations):
        df_avg_std = get_avg_std(df_predictions, avg_alias, std_alias)
        df_avg = df_avg_std.drop(std_alias).rename({avg_alias: f"{avg_alias}_{i}"})
        df_avgs = df_avgs.hstack(df_avg)
        df_y_avg_std = df_avg_std.with_columns(df.select(y_col))
        X, _, y = get_dataframe_output(
            df_y_avg_std=df_y_avg_std,
            avg_alias=avg_alias,
            std_alias=std_alias,
            y_col=y_col,
        )
        rmse = get_rmse(y=y, y_pred=X)
        rmse_list.append(rmse)
        df_predictions = remove_outliers(
            df=df_predictions,
            df_avg_std=df_avg_std,
            std_factor=std_factor,
            avg_alias=avg_alias,
            std_alias=std_alias,
        )
    make_bar_plot(
        cfg=cfg,
        x_axis=range(iterations),
        y_axis=rmse_list,
        x_label="Iterations",
        y_label="RMSE (eV)",
        title=f"RMSE vs Iterations with std_factor = {std_factor}",
    )
    df_y_avg = df_avgs.hstack(df.select(y_col))
    make_multiclass_parity_plot(cfg=cfg, df=df_y_avg, y_col=y_col)
    X, std, y = get_dataframe_output(
        df_y_avg_std=df_y_avg_std,
        avg_alias=avg_alias,
        std_alias=std_alias,
        y_col=y_col,
    )
    lower = X - std
    upper = X + std
    bins = 6
    binned_indices = get_binned_indices(data=X.squeeze(), bins=bins)
    fsc = get_fsc_metric(binned_indices, y, lower, upper)
    rmse = get_rmse(y=y, y_pred=X)
    inset = f"RMSE: {rmse:.3f} \n FSC (bins={bins}): {fsc:.3f}"
    make_parity_plot(
        cfg=cfg,
        x_axis=X,
        y_axis=y,
        yerr=std.squeeze(),
        x_label=f"{avg_alias}_{iterations - 1}",
        y_label=y_col,
        title=f"Parity Plot with std_factor = {std_factor}",
        inset=inset,
    )


def split_df(
    df: pl.DataFrame,
    holdout_indices: list[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    holdout_slice = df.filter(pl.arange(0, df.height).is_in(holdout_indices))
    df_slice = df.filter(~pl.arange(0, df.height).is_in(holdout_indices))
    return holdout_slice, df_slice


def get_dataframe_output(
    df_y_avg_std: pl.DataFrame, avg_alias: str, std_alias: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    X = df_y_avg_std[avg_alias].to_numpy().reshape(-1, 1)
    std = df_y_avg_std[std_alias].to_numpy().reshape(-1, 1)
    y = df_y_avg_std[y_col].to_numpy()
    return X, std, y


def adjust_model(model: LinearRegression, holdout: np.ndarray) -> LinearRegression:
    difference = float(holdout - model.predict(holdout))
    model.intercept_ += difference
    return model


def get_x_grid(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    min_val = float(min(min(X), min(y)))
    max_val = float(max(max(X), max(X)))
    return np.linspace(min_val, max_val, 100).reshape(-1, 1)


def shift_data(
    df_y_avg_std: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    std_alias: str,
    holdout_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    holdout_slice, df_slice = split_df(df=df_y_avg_std, holdout_indices=holdout_indices)
    holdout_means = holdout_slice.mean()
    difference = (
        holdout_means.select(y_col).item() - holdout_means.select(avg_alias).item()
    )
    X, std, y = get_dataframe_output(
        df_y_avg_std=df_slice,
        avg_alias=avg_alias,
        std_alias=std_alias,
        y_col=y_col,
    )
    X = X + difference
    return X, std, y


def linearize_data(
    df_y_avg_std: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    std_alias: str,
    holdout_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    holdout_slice, df_slice = split_df(df=df_y_avg_std, holdout_indices=holdout_indices)
    X, std, y = get_dataframe_output(
        df_y_avg_std=df_slice, avg_alias=avg_alias, std_alias=std_alias, y_col=y_col
    )
    X_holdout, _, y_holdout = get_dataframe_output(
        df_y_avg_std=holdout_slice,
        avg_alias=avg_alias,
        std_alias=std_alias,
        y_col=y_col,
    )
    model = LinearRegression().fit(X_holdout, y_holdout)
    X = model.coef_ * X + model.intercept_
    return X, std, y


def modify_data(
    linearize: bool,
    holdout_indices: list[int],
    df_y_avg_std: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    std_alias: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if linearize:
        X, std, y = linearize_data(
            df_y_avg_std=df_y_avg_std,
            y_col=y_col,
            avg_alias=avg_alias,
            std_alias=std_alias,
            holdout_indices=holdout_indices,
        )
    else:
        X, std, y = shift_data(
            df_y_avg_std=df_y_avg_std,
            y_col=y_col,
            avg_alias=avg_alias,
            std_alias=std_alias,
            holdout_indices=holdout_indices,
        )
    return X, std, y


def vary_holdout_indices(
    cfg: Config,
    num_indices: int,
    df_y_avg_std: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    std_alias: str,
    linearize: bool = False,
    bins: int = 6,
) -> None:
    rmse_list = []
    for holdout_indices in itertools.combinations(
        range(1, df_y_avg_std.height), num_indices
    ):
        X, _, y = modify_data(
            linearize=linearize,
            holdout_indices=holdout_indices,
            df_y_avg_std=df_y_avg_std,
            y_col=y_col,
            avg_alias=avg_alias,
            std_alias=std_alias,
        )
        rmse = get_rmse(y=y, y_pred=X)
        rmse_list.append(rmse)
    return rmse_list


def n_shot(
    cfg: Config,
    df_y_avg_std: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    std_alias: str,
    holdout_indices: int,
    n: int = 5,
    linearize: bool = False,
) -> None:
    X, std, y = modify_data(
        linearize=linearize,
        holdout_indices=holdout_indices,
        df_y_avg_std=df_y_avg_std,
        y_col=y_col,
        avg_alias=avg_alias,
        std_alias=std_alias,
    )
    lower = X - std
    upper = X + std
    bins = 6
    binned_indices = get_binned_indices(data=X.squeeze(), bins=bins)
    fsc = get_fsc_metric(binned_indices, y, lower, upper)
    rmse = get_rmse(y=y, y_pred=X)
    inset = f"RMSE: {rmse:.3f} \n FSC (bins={bins}): {fsc:.3f}"
    make_parity_plot(
        cfg=cfg,
        x_axis=X,
        y_axis=y,
        yerr=std.squeeze(),
        x_label=avg_alias,
        y_label=y_col,
        title=f"Parity Plot with {holdout_indices} holdouts",
        inset=inset,
    )
    if linearize:
        index_sizes = range(2, n + 1)
    else:
        index_sizes = range(1, n + 1)
    rmse_mean_list = []
    for num_indices in index_sizes:
        rmse_list = vary_holdout_indices(
            cfg=cfg,
            num_indices=num_indices,
            df_y_avg_std=df_y_avg_std,
            y_col=y_col,
            avg_alias=avg_alias,
            std_alias=std_alias,
            linearize=linearize,
        )
        rmse_mean = np.mean(rmse_list)
        rmse_mean_list.append(rmse_mean)
        make_histogram_plot(
            cfg=cfg,
            data=rmse_list,
            mean=rmse_mean,
            x_label="RMSE (eV)",
            title=f"Histogram: Bins = {bins}; Holdouts = {num_indices}",
            bins=bins,
            file_tag=f"holdouts_{num_indices}",
        )
    make_bar_plot(
        cfg=cfg,
        x_axis=index_sizes,
        y_axis=rmse_mean_list,
        x_label="Number of holdouts",
        y_label="Mean RMSE (eV)",
        title="Mean RMSE vs Number of Holdouts",
    )

import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression

from trot.config import Config
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
        df_y_avg = df_avg.with_columns(df.select(y_col))
        X, y = get_dataframe_output(
            df_y_avg=df_y_avg,
            avg_alias=f"{avg_alias}_{i}",
            y_col=y_col,
        )
        rmse = get_rmse(y=y, y_pred=X)
        rmse_list.append(rmse)
        df_avgs = df_avgs.hstack(df_avg)
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


def split_df(
    df: pl.DataFrame,
    holdout_index: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    holdout_slice = df.slice(holdout_index, 1)
    df_slice = df.filter(pl.arange(0, df.height) != holdout_index)
    return holdout_slice, df_slice


def get_dataframe_output(
    df_y_avg: pl.DataFrame, avg_alias: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    X = df_y_avg[avg_alias].to_numpy().reshape(-1, 1)
    y = df_y_avg[y_col].to_numpy()
    return X, y


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


def get_one_shot_data(
    df_y_avg: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    holdout_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    holdout_slice, df_slice = split_df(df=df_y_avg, holdout_index=holdout_index)
    difference = (
        holdout_slice.select(y_col).item() - holdout_slice.select(avg_alias).item()
    )
    X, y = get_dataframe_output(
        df_y_avg=df_slice,
        avg_alias=avg_alias,
        y_col=y_col,
    )
    X = X + difference
    return X, y


def one_shot(
    cfg: Config,
    holdout_index: int,
    df_y_avg: pl.DataFrame,
    avg_alias: str,
    y_col: str,
) -> None:
    X, y = get_one_shot_data(
        df_y_avg=df_y_avg,
        y_col=y_col,
        avg_alias=avg_alias,
        holdout_index=holdout_index,
    )
    rmse = get_rmse(y=y, y_pred=X)
    make_parity_plot(
        cfg=cfg,
        x_axis=X,
        y_axis=y,
        x_label=avg_alias,
        y_label=y_col,
        title=f"Parity Plot with sample {holdout_index} holdout",
        inset=rmse,
    )

    rmse_list = []
    for holdout_index in range(1, df_y_avg.height):
        X, y = get_one_shot_data(
            df_y_avg=df_y_avg,
            y_col=y_col,
            avg_alias=avg_alias,
            holdout_index=holdout_index,
        )
        rmse = get_rmse(y=y, y_pred=X)
        rmse_list.append(rmse)
    make_histogram_plot(
        cfg=cfg,
        data=rmse_list,
        x_label="RMSE (eV)",
        bins=6,
    )


def get_few_shot_data(
    df_y_avg: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    holdout_indices: list,
) -> tuple[np.ndarray, np.ndarray]:
    holdout_slice, df_slice = split_df(df=df_y_avg, holdout_indices=holdout_indices)
    X, y = get_dataframe_output(
        df_y_avg=df_slice,
        avg_alias=avg_alias,
        y_col=y_col,
    )
    X_holdout, y_holdout = get_dataframe_output(
        df_y_avg=holdout_slice, avg_alias=avg_alias, y_col=y_col
    )
    model = LinearRegression().fit(X_holdout, y_holdout)
    X = model.coef_ * X + model.intercept_
    return X, y


def two_shot(
    cfg: Config,
    df_y_avg: pl.DataFrame,
    y_col: str,
    avg_alias: str,
    holdout_indices: list,
) -> None:
    X, y = get_few_shot_data(
        df_y_avg=df_y_avg,
        y_col=y_col,
        avg_alias=avg_alias,
        holdout_indices=holdout_indices,
    )
    rmse = get_rmse(y=y, y_pred=X)
    make_parity_plot(
        cfg=cfg,
        x_axis=X,
        y_axis=y,
        x_label=avg_alias,
        y_label=y_col,
        title=f"Parity Plot with {len(holdout_indices)} holdouts",
    )

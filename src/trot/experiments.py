import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression

from trot.config import Config
from trot.evaluate import df_mse, ndarray_mse
from trot.visualize import make_bar_plot, make_parity_plot, make_histogram_plot


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
    df_y = df.select(y_col)
    df_predictions = df.select(pl.exclude(y_col))
    mse_list = []
    df_avgs = pl.DataFrame()
    for i in range(iterations):
        df_avg_std = get_avg_std(df_predictions, avg_alias, std_alias)
        df_avg = pl.select(df_avg_std.drop(std_alias)).rename(
            {avg_alias: f"{avg_alias}_{i}"}
        )
        df_avgs = df_avgs.hstack(df_avg)
        df_avg_std.write_parquet(f"{cfg.paths.results.iter_avg}/avg_std{i}.parquet")
        df_predictions = remove_outliers(
            df=df_predictions,
            df_avg_std=df_avg_std,
            std_factor=std_factor,
            avg_alias=avg_alias,
            std_alias=std_alias,
        )
        df_y_avg = df_avg_std.hstack(df_y)
        mse = df_mse(df=df_y_avg, y_col=y_col, pred_col=avg_alias)
        mse_list.append(mse)
    make_bar_plot(
        cfg=cfg,
        x_axis=range(iterations),
        y_axis=mse_list,
        x_label="Iterations",
        y_label=r"MSE (eV$^{2}$)",
        title=f"MSE vs Iterations with std_factor = {std_factor}",
    )
    df_y_avg = df_avgs.hstack(df_y)
    make_parity_plot(cfg=cfg, df=df_y_avg, y_col=y_col)


def split_df(
    df: pl.DataFrame,
    holdout_index: int,
    y_col: str,
) -> tuple[np.ndarray, pl.DataFrame]:
    holdout_slice = df.slice(holdout_index, holdout_index + 1)[y_col][0]
    holdout = np.array([[holdout_slice]])
    df_slice = df.slice(1, df.height - 1)
    return holdout, df_slice


def get_dataframe_output(
    df_y: pl.DataFrame, df_avg: pl.DataFrame, avg_alias: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    df_y_avg = df_y.hstack(df_avg)
    X = df_y_avg[y_col].to_numpy().reshape(-1, 1)
    y = df_y_avg[avg_alias].to_numpy()
    return X, y


def adjust_intercept(model: LinearRegression, holdout: np.ndarray) -> LinearRegression:
    difference = holdout - model.predict(holdout)
    model.intercept_ += difference
    model.intercept_ = float(model.intercept_)
    return model


def get_x_grid(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    min_val = float(min(min(X), min(y)))
    max_val = float(max(max(X), max(X)))
    return np.linspace(min_val, max_val, 100).reshape(-1, 1)


def one_shot(
    cfg: Config,
    holdout_index: int,
    df: pl.DataFrame,
    avg_alias: str,
    y_col: str,
) -> None:
    mse_list = []
    for holdout_index in range(1, df.height):
        holdout, df_slice = split_df(df, holdout_index, y_col)
        df_avg = get_avg_std(
            df=df_slice,
            avg_alias="average",
            std_alias="std",
        ).drop("std")
        X, y = get_dataframe_output(
            df_y=df_slice.select(y_col),
            df_avg=df_avg,
            avg_alias=avg_alias,
            y_col=y_col,
        )
        model = LinearRegression()
        model.fit(X=X, y=y)
        model = adjust_intercept(model, holdout)
        y_pred = model.predict(X)
        mse = ndarray_mse(y, y_pred)
        mse_list.append(mse)
    make_histogram_plot(
        cfg=cfg,
        data=mse_list,
        x_label=r"MSE (eV$^{2}$)",
        bins=6,
    )
    x_grid = get_x_grid(X=X, y=y)
    make_parity_plot(
        cfg=cfg,
        x_axis=x_grid,
        y_axis=model.predict(x_grid),
        x_label=avg_alias,
        y_label=y_col,
        title=f"Parity Plot with sample {holdout_index} holdout",
    )
    y_pred = model.predict(X)


def two_shot(
    holdout_indices: list,
) -> None:
    pass

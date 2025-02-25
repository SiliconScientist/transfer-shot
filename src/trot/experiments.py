import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression

from trot.config import Config
from trot.evaluate import get_mse
from trot.visualize import make_bar_plot, make_parity_plot


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
        mse = get_mse(df=df_y_avg, y_col=y_col, pred_col=avg_alias)
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


def one_shot(
    cfg: Config,
    holdout_index: int,
    df: pl.DataFrame,
    avg_alias: str,
    y_col: str,
) -> None:
    holdout_slice = df.slice(holdout_index, holdout_index + 1)[y_col][0]
    holdout = np.array([[holdout_slice]])
    df = df.slice(1, df.height - 1)
    df_avg = get_avg_std(
        df=df,
        avg_alias="average",
        std_alias="std",
    ).drop("std")
    df_y = df.select(y_col)
    df_y_avg = df_y.hstack(df_avg)
    X = df_y_avg[y_col].to_numpy().reshape(-1, 1)
    y = df_y_avg[avg_alias].to_numpy()
    model = LinearRegression()
    model.fit(X=X, y=y)
    difference = holdout - model.predict(holdout)
    model.intercept_ += difference
    model.intercept_ = float(model.intercept_)
    min_val = float(min(min(X), min(y)))
    max_val = float(max(max(X), max(X)))
    parity_line = np.linspace(min_val, max_val, 100).reshape(-1, 1)
    y_avg = model.predict(parity_line)
    make_parity_plot(
        cfg=cfg,
        model=model,
        parity_line=parity_line,
        x_axis=X,
        y_axis=y_avg,
        x_label=avg_alias,
        y_label=y_col,
        title=f"Parity Plot with sample {holdout_index} holdout",
    )


def two_shot(
    holdout_indices: list,
) -> None:
    pass

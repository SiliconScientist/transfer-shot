import polars as pl

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

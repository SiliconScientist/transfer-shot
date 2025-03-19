import toml
import polars as pl


from trot.config import Config
from trot.processing import get_data
from trot.experiments import iterative_averages, n_shot, get_avg_std
from trot.visualize import make_multiclass_parity_plot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    # make_multiclass_parity_plot(cfg=cfg, df=df, y_col="DFT")
    # iterative_averages(cfg=cfg, df=df, y_col="DFT")
    y_col = "DFT"
    df_predictions = df.select(pl.exclude(y_col))
    df_avg_std = get_avg_std(
        df=df_predictions,
        avg_alias="average",
        std_alias="std",
    )
    df_y_avg_std = df_avg_std.with_columns(df.select(y_col))
    n_shot(
        cfg=cfg,
        df_y_avg_std=df_y_avg_std,
        holdout_indices=[0, 5, 10, 15, 20],
        avg_alias="average",
        std_alias="std",
        y_col="DFT",
        linearize=True,
    )


if __name__ == "__main__":
    main()

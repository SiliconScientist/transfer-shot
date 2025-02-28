import toml
import polars as pl


from trot.config import Config
from trot.processing import get_data
from trot.experiments import few_shot, get_avg_std


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    y_col = "DFT"
    df_predictions = df.select(pl.exclude(y_col))
    df_avg = get_avg_std(
        df=df_predictions,
        avg_alias="average",
        std_alias="std",
    ).drop("std")
    df_y_avg = df_avg.with_columns(df.select(y_col))
    few_shot(
        cfg=cfg,
        df_y_avg=df_y_avg,
        holdout_indices=[5, 17],
        avg_alias="average",
        y_col="DFT",
        linearize=True,
    )


if __name__ == "__main__":
    main()

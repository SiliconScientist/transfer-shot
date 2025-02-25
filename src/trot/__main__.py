import toml
import polars as pl
import numpy as np


from trot.config import Config
from trot.processing import get_data

from trot.experiments import get_avg_std, one_shot
from trot.visualize import clean_column_name


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    df = df.rename({col: clean_column_name(col) for col in df.columns})
    one_shot(
        cfg=cfg,
        df=df,
        holdout_index=15,
        avg_alias="average",
        y_col="DFT",
    )


if __name__ == "__main__":
    main()

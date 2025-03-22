import toml
import polars as pl


from trot.config import Config
from trot.processing import get_data, get_holdout_split
from trot.experiments import n_shot, get_avg_std


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    n_shot(cfg=cfg, df=df)


if __name__ == "__main__":
    main()

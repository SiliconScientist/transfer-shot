import toml
import polars as pl


from trot.config import Config
from trot.processing import get_data
from trot.experiments import n_shot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg, holdout_set=False)
    n_shot(
        cfg=cfg,
        df=df,
        max_samples=cfg.max_samples,
        linearize=cfg.linearize,
    )
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()

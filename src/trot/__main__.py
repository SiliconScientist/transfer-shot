import toml


from trot.config import Config
from trot.processing import get_data
from trot.experiments import one_shot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    one_shot(
        cfg=cfg,
        df=df,
        holdout_index=15,
        avg_alias="average",
        y_col="DFT",
    )


if __name__ == "__main__":
    main()

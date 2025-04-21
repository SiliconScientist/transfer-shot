import toml


from trot.config import Config
from trot.processing import get_data
from trot.experiments import n_shot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg, holdout_set=False)
    print(df)
    # df_holdout = get_data(cfg, holdout_set=True)
    # n_shot(
    #     cfg=cfg,
    #     df=df,
    #     df_holdout=df_holdout,
    #     max_samples=cfg.max_samples,
    #     linearize=cfg.linearize,
    # )
    # print("Experiment completed successfully!")


if __name__ == "__main__":
    main()

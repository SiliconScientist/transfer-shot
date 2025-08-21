import toml
import polars as pl


from trot.config import Config
from trot.processing import get_data
from trot.experiments import get_recommendation, greedy_cost, n_shot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg, holdout_set=False)
    print(f"Dataframe shape: {df.shape}")
    # df_holdout = get_data(cfg, holdout_set=True)
    # n_shot(cfg, df)
    # cost_fn = greedy_cost
    # recommendation_index = get_recommendation(
    #     cfg=cfg,
    #     df=df,
    #     cost_fn=cost_fn,
    #     df_holdout=df_holdout,
    # )
    # print(f"Recommended index: {recommendation_index}")


if __name__ == "__main__":
    main()

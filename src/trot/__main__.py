import toml
import polars as pl
from functools import partial


from trot.config import Config
from trot.processing import get_data
from trot.experiments import get_recommendation, greedy_cost, n_shot
from trot.visualize import make_summary_figure


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg, holdout_set=False)
    # df = df.drop("DIMENET")
    # df = df.drop("SCHNET")
    # df = df.drop("PAINN")
    # df = df.drop("SCN")
    # df = df.drop("GEMNET")
    # df_subset = df.sample(n=0, seed=42)
    # df_holdout = get_data(cfg, holdout_set=True)
    results = n_shot(
        cfg,
        df,
        max_samples=cfg.max_samples,
        linearize=cfg.linearize,
        seed=cfg.random_seed,
    )
    make_summary_figure(
        cfg=cfg,
        df=df,
        results=results,
        filename="data/results/visualizations/n_shot_summary.png",
    )
    # cost_fn = partial(greedy_cost, alpha=0.75)
    # recommendation_index = get_recommendation(
    #     cfg=cfg,
    #     df=df,
    #     cost_fn=cost_fn,
    #     df_holdout=df_holdout,
    # )
    # print(f"Recommended index: {recommendation_index}")


if __name__ == "__main__":
    main()

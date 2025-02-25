import toml


from trot.config import Config
from trot.processing import get_data

from trot.experiments import iterative_averages
from trot.visualize import clean_column_name


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    df = df.rename({col: clean_column_name(col) for col in df.columns})
    y_col = "DFT"
    iterative_averages(cfg=cfg, df=df, y_col=y_col)


if __name__ == "__main__":
    main()

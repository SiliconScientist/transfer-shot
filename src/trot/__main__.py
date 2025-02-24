import toml


from trot.config import Config
from trot.processing import get_data
from trot.visualize import clean_column_name, make_parity_plot


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    df = df.rename({col: clean_column_name(col) for col in df.columns})
    make_parity_plot(df, y_col="DFT")


if __name__ == "__main__":
    main()

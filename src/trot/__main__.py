import toml


from trot.config import Config
from trot.processing import get_data
from trot.evaluate import get_mse


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    mse = get_mse(df=df, y_col="DFT", pred_col="DimeNet++-S2EF-OC20-All")
    print(f"MSE: {mse}")


if __name__ == "__main__":
    main()

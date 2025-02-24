import toml


from trot.config import Config
from trot.processing import get_data


def main():
    cfg = Config(**toml.load("config.toml"))
    df = get_data(cfg)
    print(df)


if __name__ == "__main__":
    main()

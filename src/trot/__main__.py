import toml

from trot.config import Config
from trot.dataset import get_dataloader


def main():
    cfg = Config(**toml.load("config.toml"))
    loader = get_dataloader(cfg)
    print(loader)


if __name__ == "__main__":
    main()

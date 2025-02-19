import toml
import os
from trot.config import Config
from trot.processing import process_data
from fairchem.core.datasets import LmdbDataset


def main():
    cfg = Config(**toml.load("config.toml"))
    if os.path.exists(cfg.paths.processed):
        dataset = LmdbDataset({"src": str(cfg.paths.processed)})
    else:
        process_data(cfg)
        dataset = LmdbDataset({"src": str(cfg.paths.processed)})
    print(dataset)


if __name__ == "__main__":
    main()

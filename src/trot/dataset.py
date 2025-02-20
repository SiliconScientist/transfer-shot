import os
from fairchem.core.datasets import LmdbDataset
from torch_geometric.loader import DataLoader

from trot.config import Config
from trot.processing import process_data


def get_dataloader(cfg: Config):
    if os.path.exists(cfg.paths.processed.graphs):
        dataset = LmdbDataset({"src": str(cfg.paths.processed.graphs)})
    else:
        process_data(cfg)
        dataset = LmdbDataset({"src": str(cfg.paths.processed.graphs)})
    loader = DataLoader(dataset, batch_size=cfg.loader.batch_size, shuffle=True)
    return loader

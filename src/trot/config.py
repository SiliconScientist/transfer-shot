from pathlib import Path

from pydantic import BaseModel


class Raw(BaseModel):
    adsorbed: Path
    bare: Path


class Processed(BaseModel):
    graphs: Path
    extxyz: Path


class DataLoader(BaseModel):
    batch_size: int


class Paths(BaseModel):
    raw: Raw
    processed: Processed


class Config(BaseModel):
    random_seed: int
    loader: DataLoader
    paths: Paths

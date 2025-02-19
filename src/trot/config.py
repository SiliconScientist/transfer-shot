from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    raw: Path


class Config(BaseModel):
    random_seed: int
    paths: Paths

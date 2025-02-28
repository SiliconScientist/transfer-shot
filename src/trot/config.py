from pathlib import Path

from pydantic import BaseModel


class Raw(BaseModel):
    adsorbed: Path
    bare: Path


class Processed(BaseModel):
    predictions: Path


class Results(BaseModel):
    adjusted_parity_plot: Path
    iter_avg_parity_plot: Path
    iter_avg: Path
    bar_plot: Path
    hist_plot: Path


class Paths(BaseModel):
    raw: Raw
    processed: Processed
    results: Results


class Config(BaseModel):
    random_seed: int
    paths: Paths

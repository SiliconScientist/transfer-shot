from pathlib import Path

from pydantic import BaseModel


class Raw(BaseModel):
    adsorbed: Path
    bare: Path


class Processed(BaseModel):
    predictions: Path


class Results(BaseModel):
    visualizations: Path
    parity_plot: Path
    multiclass_parity_plot: Path
    iter_avg: Path
    bar_plot: Path
    hist_plot: Path


class Processing(BaseModel):
    gas_phase_adsorbate: str
    num_atoms_adsorbed: int


class Paths(BaseModel):
    raw: Raw
    processed: Processed
    results: Results


class Config(BaseModel):
    random_seed: int
    max_samples: int
    linearize: bool
    removal_iterations: int
    std_factor: int
    y_key: str
    paths: Paths
    processing: Processing

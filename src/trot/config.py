from pathlib import Path
from pydantic import BaseModel


class Raw(BaseModel):
    adsorbed: Path
    bare: Path


class Processed(BaseModel):
    predictions: Path
    holdout_predictions: Path


class Results(BaseModel):
    visualizations: Path
    parity_plot: Path
    bar_plot: Path


class Processing(BaseModel):
    gas_phase_adsorbate: str
    num_atoms_adsorbed: int


class Paths(BaseModel):
    raw: Raw
    processed: Processed
    results: Results


class Config(BaseModel):
    random_seed: int
    dev_run: bool
    cpu: bool
    target: float
    remove_high_variance: bool
    variance_threshold: float
    max_samples: int
    linearize: bool
    removal_iterations: int
    std_factor: int
    y_key: str
    paths: Paths
    processing: Processing

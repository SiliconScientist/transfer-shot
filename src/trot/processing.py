import re
import numpy as np
import polars as pl
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from tqdm import tqdm

from trot.config import Config
from trot.model import get_calculator, set_calculators

MODEL_NAMES = [
    "DimeNet++-S2EF-OC20-All",
    "SchNet-S2EF-OC20-All",
    "PaiNN-S2EF-OC20-All",
    "SCN-S2EF-OC20-All+MD",
    "GemNet-dT-S2EF-OC20-All",
]


def get_potential_energies(atoms_list: list[Atoms], default_energy: float = np.nan):
    energies = []
    for atoms in atoms_list:
        if atoms.calc is None:
            atoms.calc = SinglePointCalculator(atoms, energy=default_energy)
        energy = atoms.get_potential_energy()
        energies.append(energy)
    return energies


def get_model_predictions(cfg: Config, atoms_list: list[Atoms]) -> pl.DataFrame:
    adsorption_energies = {}
    for name in MODEL_NAMES:
        calc = get_calculator(cfg=cfg, name=name)
        atoms_list_copy = set_calculators(atoms_list, calc)
        energies = [
            atoms.get_potential_energy()
            for atoms in tqdm(
                atoms_list_copy, desc=f"Energies ({name})", total=len(atoms_list_copy)
            )
        ]
        adsorption_energies[name] = energies
    return pl.DataFrame(adsorption_energies)


def clean_column_name(name):
    match = re.match(r"^([A-Za-z0-9]+)", name)
    return match.group(1).upper() if match else name


def build_df(
    cfg: Config, atoms_list: list[Atoms], energies: list[float]
) -> pl.DataFrame:
    if cfg.dev_run:
        atoms_list = atoms_list[:5]
        energies = energies[:5]
    df_y = pl.DataFrame({cfg.y_key: energies})
    df_predictions = get_model_predictions(cfg, atoms_list)
    df = pl.concat([df_y, df_predictions], how="horizontal")
    df = df.rename({col: clean_column_name(col) for col in df.columns})
    return df


def get_predictions(cfg: Config) -> pl.DataFrame:
    atoms_list = read(filename=cfg.paths.raw.adsorbed, index=":")
    energies = get_potential_energies(atoms_list=atoms_list)
    df = build_df(cfg=cfg, atoms_list=atoms_list, energies=energies)
    df.write_parquet(cfg.paths.processed.predictions)


def remove_high_variance_samples(
    cfg: Config, df: pl.DataFrame, variance_threshold: float = 1.0
) -> pl.DataFrame:
    prediction_cols = [col for col in df.columns if col != cfg.y_key]
    df = df.with_columns(pl.concat_list(prediction_cols).list.std().alias("std_dev"))
    df_filtered = df.filter(pl.col("std_dev") <= variance_threshold).drop("std_dev")
    return df_filtered


def get_data(cfg: Config, holdout_set: bool) -> pl.DataFrame:
    if cfg.paths.processed.predictions.exists():
        df = pl.read_parquet(cfg.paths.processed.predictions)
    else:
        get_predictions(cfg)
        df = pl.read_parquet(cfg.paths.processed.predictions)
    if holdout_set:
        df = pl.read_parquet(cfg.paths.processed.holdout_predictions)
    if cfg.remove_high_variance:
        df = remove_high_variance_samples(
            cfg=cfg, df=df, variance_threshold=cfg.variance_threshold
        )
    return df


def df_to_numpy(
    df: pl.DataFrame, y_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_X = df.select(pl.exclude(y_col))
    X = df_X.to_numpy()
    y = df[y_col].to_numpy()
    return X, y


def get_holdout_split(
    X: np.ndarray, y: np.ndarray, holdout_indices: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[list(holdout_indices)] = True
    return X[~mask], y[~mask], X[mask], y[mask]

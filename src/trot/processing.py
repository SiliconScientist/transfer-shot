import re
import numpy as np
import polars as pl
from ase.db import connect
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pathlib import Path
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

# Units: eV
GAS_PHASE_ENERGIES = {
    "H2": -6.74815624,
    "O2": -9.848396,
}


def get_atoms_list(raw_path: Path) -> list[Atoms]:
    if Path(raw_path).suffix == ".extxyz":
        atoms_list = []
        with open(raw_path, "r") as f:
            for line in f:
                if line.startswith("Atoms"):
                    atoms = Atoms(line.split()[1])
                    atoms_list.append(atoms)
        return atoms_list
    elif Path(raw_path).suffix == ".db":
        db = connect(raw_path)
        atoms_list = []
        for row in db.select():
            atoms = row.toatoms()
            atoms_list.append(atoms)
        return atoms_list
    else:
        raise ValueError(f"Unsupported file format: {raw_path.suffix}")


def get_potential_energies(atoms_list: list[Atoms], default_energy: float = np.nan):
    energies = []
    for atoms in atoms_list:
        if atoms.calc is None:
            atoms.calc = SinglePointCalculator(atoms, energy=default_energy)
        energy = atoms.get_potential_energy()
        energies.append(energy)
    return energies


def filter_for_host(
    atoms_list: list[Atoms],
    host_atom: str,
    host_criteria: int = 10,
) -> list[Atoms]:
    # The host material will be the majority of the surface (i.e. > 10 atoms)
    filtered_atoms = [
        atoms
        for atoms in atoms_list
        if atoms.get_chemical_symbols().count(host_atom) > host_criteria
    ]
    return filtered_atoms


def get_adsorption_energy(adsorbed: float, bare: float, adsorbate: float) -> float:
    return adsorbed - bare - adsorbate


def get_adsorption_energies(
    bare_surface: list[Atoms],
    adsorbed_surface: list[Atoms],
    adsorbate: float,
) -> list[float]:
    bare_adsorbed = zip(bare_surface, adsorbed_surface)
    adsorption_energies = []
    for bare, adsorbed in bare_adsorbed:
        adsorption_energy = get_adsorption_energy(
            adsorbed=adsorbed.get_potential_energy(),
            bare=bare.get_potential_energy(),
            adsorbate=adsorbate,
        )
        adsorption_energies.append(adsorption_energy)
    return adsorption_energies


def process_data(cfg: Config) -> tuple[list[float], list[Atoms]]:
    bare_list = get_atoms_list(cfg.paths.raw.bare)
    adsorbed_list = get_atoms_list(cfg.paths.raw.adsorbed)
    bare = filter_for_host(atoms_list=bare_list, host_atom="Ag")
    adsorbed = filter_for_host(atoms_list=adsorbed_list, host_atom="Ag")
    energies = get_adsorption_energies(
        bare_surface=bare,
        adsorbed_surface=adsorbed,
        adsorbate=GAS_PHASE_ENERGIES["O2"] / 2,
    )
    return energies, adsorbed


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
        atoms_list = atoms_list[:1]
        energies = energies[:1]
    df_y = pl.DataFrame({"DFT": energies})
    df_predictions = get_model_predictions(cfg, atoms_list)
    df = pl.concat([df_y, df_predictions], how="horizontal")
    df = df.rename({col: clean_column_name(col) for col in df.columns})
    return df


def get_predictions(cfg: Config) -> pl.DataFrame:
    if cfg.paths.raw.bare.exists():
        energies, atoms_list = process_data(cfg)
    else:
        atoms_list = read(filename=cfg.paths.raw.adsorbed, index=":")
        energies = get_potential_energies(atoms_list=atoms_list)
    df = build_df(cfg=cfg, atoms_list=atoms_list, energies=energies)
    df.write_parquet(cfg.paths.processed.predictions)


def remove_high_variance_samples(
    df: pl.DataFrame, variance_threshold: float = 1.0
) -> pl.DataFrame:
    prediction_cols = [col for col in df.columns if col != "DFT"]
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
            df=df, variance_threshold=cfg.variance_threshold
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

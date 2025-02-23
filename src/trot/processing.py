from ase.db import connect
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pathlib import Path

from trot.config import Config

H2_GAS_PHASE_ENERGY: float = -6.74815624  # eV


def get_atoms_list(raw_path: Path) -> list[Atoms]:
    db = connect(raw_path)
    atoms_list = []
    for row in db.select():
        atoms = row.toatoms()
        atoms_list.append(atoms)
    return atoms_list


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
        adsorbate=H2_GAS_PHASE_ENERGY / 2,
    )
    return energies, adsorbed

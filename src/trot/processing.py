import lmdb
import torch
import pickle
from ase.db import connect
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pathlib import Path
from fairchem.core.preprocessing import AtomsToGraphs
from tqdm import tqdm

from trot.config import Config

H2_GAS_PHASE_ENERGY: float = -6.74815624  # eV


def get_atoms_list(raw_path: Path) -> list[Atoms]:
    db = connect(raw_path)
    atoms_list = []
    for row in db.select():
        atoms = row.toatoms()
        atoms_list.append(atoms)
    return atoms_list


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


def set_energies(atoms_list: list[Atoms], energies: list[float]) -> list[Atoms]:
    reset_atoms_list = []
    for atoms, energy in zip(atoms_list, energies):
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        reset_atoms_list.append(atoms)
    return reset_atoms_list


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


def process_data(cfg: Config) -> None:
    bare_surfaces = get_atoms_list(cfg.paths.raw.bare)
    adsorbed_surfaces = get_atoms_list(cfg.paths.raw.adsorbed)
    adsorption_energies = get_adsorption_energies(
        bare_surface=bare_surfaces,
        adsorbed_surface=adsorbed_surfaces,
        adsorbate=H2_GAS_PHASE_ENERGY / 2,
    )
    atoms_list = set_energies(adsorbed_surfaces, adsorption_energies)
    filtered_atoms = filter_for_host(
        atoms_list, host_atom="Ag"
    )  # TODO: put the host_atom in the config file
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,  # False for test data
        r_forces=False,  # False for test data
        r_distances=False,
        r_fixed=True,
    )
    data_objects = a2g.convert_all(filtered_atoms, disable_tqdm=True)
    tags = filtered_atoms[0].get_tags()
    db = lmdb.open(
        str(cfg.paths.processed.graphs),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
        # assign sid
        data.sid = torch.LongTensor([0])
        # assign fid
        data.fid = torch.LongTensor([fid])
        # assign tags, if available
        data.tags = torch.LongTensor(tags)
        # Filter data if necessary
        # FAIRChem filters adsorption energies > |10| eV and forces > |50| eV/A
        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            # print("no neighbors", traj_path)
            continue
        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
    txn.commit()
    db.sync()
    db.close()

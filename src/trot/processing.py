import lmdb
import torch
import pickle
from ase.db import connect
from ase.atoms import Atoms
from pathlib import Path
from fairchem.core.preprocessing import AtomsToGraphs
from tqdm import tqdm

from trot.config import Config


def get_atoms_list(raw_path: Path) -> list[Atoms]:
    db = connect(raw_path)
    atoms_list = []
    for row in db.select():
        atoms = row.toatoms()
        atoms_list.append(atoms)
    return atoms_list


def process_data(cfg: Config) -> None:
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,  # False for test data
        r_forces=False,  # False for test data
        r_distances=False,
        r_fixed=True,
    )
    atoms_list = get_atoms_list(cfg.paths.raw)
    data_objects = a2g.convert_all(atoms_list, disable_tqdm=True)
    tags = atoms_list[0].get_tags()
    db = lmdb.open(
        str(cfg.paths.processed),
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

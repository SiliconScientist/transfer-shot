from ase.db import connect
from pathlib import Path


def get_atoms_list(raw_path: Path):
    db = connect(raw_path)
    atoms_list = []
    for row in db.select():
        atoms = row.toatoms()
        atoms_list.append(atoms)
    return atoms_list

from __future__ import annotations

from ase import Atoms
import json
from pathlib import Path

from ase.db import connect
from ase.io import read, write
import numpy as np
from fairchem.core.datasets import AseDBDataset
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from fairchem.core.components.calculate.relaxation_runner import RelaxationRunner

# Python 3.11+ has tomllib in stdlib; older needs tomli.
try:
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def decode_ndarray(field):
    """
    Decode an ASE/MSON-style "__ndarray__" field into a numpy array
    with the right shape.
    """
    shape, dtype, data = field["__ndarray__"]
    arr = np.array(data, dtype=dtype)
    return arr.reshape(shape)


def ensure_ase_db(src_path: str, dev_run: bool) -> str:
    """
    Convert an extxyz file into a temporary ASE DB so RelaxationRunner can load it.
    """
    src = Path(src_path)
    if src.suffix != ".extxyz":
        if not dev_run:
            return str(src)
        db_path = src.parent / f"{src.stem}.dev.db"
        with connect(src) as db:
            try:
                row = next(db.select())
            except StopIteration as exc:
                raise ValueError(f"No rows found in {src}") from exc
            db.write(row.toatoms())
        return str(db_path)

    if dev_run:
        atoms_list = [read(filename=str(src), index=0)]
    else:
        atoms_list = read(filename=str(src), index=":")
    if not atoms_list:
        raise ValueError(f"No frames found in {src}")

    # Avoid clobbering an existing .db by writing to a derived filename.
    db_suffix = ".dev.db" if dev_run else ".db"
    db_path = src.parent / f"{src.stem}{db_suffix}"
    with connect(db_path) as db:
        for atoms in atoms_list:
            db.write(atoms)
    return str(db_path)


def load_initial_energies(src_path: str, dev_run: bool):
    src = Path(src_path)
    atoms_list = []

    if src.suffix == ".db":
        with connect(str(src)) as db:
            rows = list(db.select(limit=1)) if dev_run else list(db.select())
            atoms_list = [row.toatoms() for row in rows]
    else:
        images = read(str(src), index=0 if dev_run else ":")
        atoms_list = images if isinstance(images, list) else [images]

    E0 = {}
    for i, atoms in enumerate(atoms_list):
        key = (
            atoms.info.get("id") or atoms.info.get("name") or atoms.info.get("uid") or i
        )
        e = atoms.info.get("energy") or atoms.info.get("E")
        if e is None and atoms.calc is not None:
            try:
                e = atoms.get_potential_energy()
            except Exception:
                pass
        E0[key] = e

    return E0, atoms_list


def load_config(config_path: str | Path = "config.toml") -> dict:
    """
    Load run configuration from a TOML file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Create it (see example below) or set CONFIG_TOML env var."
        )
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> None:
    cfg = load_config("config.toml")

    # Keep defaults identical to argparse defaults in your original script
    src_path = cfg.get("src", "relaxed_mamun_oh.extxyz")
    output_path = cfg.get("output", None)
    dev_run = bool(cfg.get("dev_run", False))
    device = cfg.get("device", "cuda")
    mlip_path = cfg.get("mlip_path", None)

    config = {"src": ensure_ase_db(src_path, dev_run)}
    dataset = AseDBDataset(config=config)
    predictor = load_predict_unit(
        path=mlip_path,
        device=device,
    )
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    runner = RelaxationRunner(
        calculator=calc,
        input_data=dataset,
        calculate_properties=[],
    )
    results = runner.calculate()
    E0_map, atoms_in = load_initial_energies(src_path, dev_run)

    relaxed_atoms = []
    for i, r in enumerate(results):
        blob = r["atoms"]
        atoms_dict = json.loads(blob["atoms_json"])
        atoms = Atoms(
            numbers=decode_ndarray(atoms_dict["numbers"]),
            positions=decode_ndarray(atoms_dict["positions"]),
            cell=decode_ndarray(atoms_dict["cell"]),
            pbc=decode_ndarray(atoms_dict["pbc"]),
        )

        # Determine the matching key the same way we did for inputs
        a_in = atoms_in[i]
        key = a_in.info.get("id") or a_in.info.get("name") or a_in.info.get("uid") or i
        e0 = E0_map.get(key)

        # Store initial energy as metadata on the relaxed structure
        if e0 is not None:
            atoms.info["energy"] = float(e0)

        # Optional: also preserve original id/name so you can join later
        if "id" in a_in.info:
            atoms.info["id"] = a_in.info["id"]
        if "name" in a_in.info:
            atoms.info["name"] = a_in.info["name"]

        relaxed_atoms.append(atoms)

    write(cfg.get("output", None), relaxed_atoms, format="extxyz")


if __name__ == "__main__":
    main()

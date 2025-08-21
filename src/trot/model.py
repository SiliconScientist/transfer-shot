import torch.nn as nn
from ase.atoms import Atoms
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

from trot.config import Config


def get_calculator(
    cfg: Config,
    name: str,
    local_cache: str = "/tmp/fairchem_checkpoints/",
) -> nn.Module:
    checkpoint_path = model_name_to_local_file(
        model_name=name,
        local_cache=local_cache,
    )
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        seed=cfg.random_seed,
        cpu=cfg.cpu,
        only_output=["energy"],
    )
    calc.trainer.model.to("cuda")
    return calc


def set_calculators(atoms_list: list[Atoms], calc: OCPCalculator) -> list[Atoms]:
    atoms_list_copy = [atoms.copy() for atoms in atoms_list]
    for atoms in atoms_list_copy:
        atoms.calc = calc
    return atoms_list_copy

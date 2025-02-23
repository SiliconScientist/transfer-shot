import torch.nn as nn
from ase.atoms import Atoms
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator


def get_calculator(
    name: str,
    local_cache: str = "/tmp/fairchem_checkpoints/",
) -> nn.Module:
    checkpoint_path = model_name_to_local_file(
        model_name=name,
        local_cache=local_cache,
    )
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
    )
    return calc


def set_calculators(atoms_list: list[Atoms], calc: OCPCalculator) -> list:
    for atoms in atoms_list:
        atoms.calc = calc
    return atoms_list

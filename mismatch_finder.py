import numpy as np
from ase.io import read, write
from typing import Iterable, List, Optional
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.singlepoint import SinglePointCalculator


def _min_image_displacements_pmg(
    struct_a, struct_b, subset: Optional[Iterable[int]] = None
):
    """Per-site min-image displacements (Å) using struct_b lattice."""
    if len(struct_a) != len(struct_b):
        raise ValueError("Structures must have the same number of sites.")
    if subset is None:
        subset = range(len(struct_a))

    f_a = struct_a.frac_coords
    f_b = struct_b.frac_coords
    lat = struct_b.lattice.matrix  # 3x3

    # (Optional) quick species sanity check:
    for i in subset:
        if struct_a[i].specie != struct_b[i].specie:
            raise ValueError(
                f"Species mismatch at site {i}: {struct_a[i].specie} vs {struct_b[i].specie}"
            )

    disps = []
    for i in subset:
        df = f_b[i] - f_a[i]
        df -= np.round(df)  # wrap to [-0.5, 0.5)
        d_cart = df @ lat
        disps.append(np.linalg.norm(d_cart))
    return np.asarray(disps)


def compare_ase_lists_for_movement(
    ase_list_ref: List,  # e.g., DFT-relaxed
    ase_list_test: List,  # e.g., MLIP-relaxed
    tol: float = 0.35,  # Å: “too much movement”
    subset: Optional[
        Iterable[int]
    ] = None,  # indices to check (e.g., adsorbate only); default = all sites
    rule: str = "max",  # "max" or "any" or "fraction"
) -> List[int]:
    """
    Returns list indices where the pair is NOT the same by the movement rule.

    Rules:
      - "max": not same if max per-site displacement > tol
      - "any": not same if any site displacement > tol (same as "max" logically)
      - "fraction": set tol as fraction threshold in [0,1): not same if fraction of sites with disp>0.2Å exceeds tol
                    (use together with a fixed 'inner_tol' below)
    """
    if len(ase_list_ref) != len(ase_list_test):
        raise ValueError("Lists must be the same length.")

    adaptor = AseAtomsAdaptor()
    bad_idxs = []

    # If using "fraction", define the *inner* movement cutoff and interpret tol as fraction.
    inner_tol = 0.20  # Å cutoff for counting “moved” sites when using the fraction rule

    for idx, (a_ref, a_test) in enumerate(zip(ase_list_ref, ase_list_test)):
        s_ref = adaptor.get_structure(a_ref)
        s_test = adaptor.get_structure(a_test)

        disps = _min_image_displacements_pmg(s_ref, s_test, subset=subset)

        if rule == "max" or rule == "any":
            moved = np.max(disps) > tol
        elif rule == "fraction":
            frac = np.mean(disps > inner_tol)
            moved = frac > tol
        else:
            raise ValueError(f"Unknown rule: {rule}")

        if moved:
            bad_idxs.append(idx)

    return bad_idxs


dft_atoms_list = read("data/raw/relaxed_mamun_oh.extxyz", index=":")
energy_list = [atoms.get_potential_energy() for atoms in dft_atoms_list]

mlip_atoms_list = read("data/raw/uma_relaxed_mamun_oh.traj", index=":")
print(len(dft_atoms_list), len(mlip_atoms_list))
atoms_list = []
for atoms, energy in zip(mlip_atoms_list, energy_list):
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms_list.append(atoms)
# DFT and MLIP lists (same length, same ordering)
pairs_bad = compare_ase_lists_for_movement(
    dft_atoms_list, mlip_atoms_list, tol=0.35, rule="max"
)
# Obtain the inverse pairs (the good ones)
pairs_good = [i for i in range(len(dft_atoms_list)) if i not in pairs_bad]

bad_atoms_list = [atoms_list[i] for i in pairs_bad]
good_atoms_list = [atoms_list[i] for i in pairs_good]

write("data/processed/bad_pairs.traj", bad_atoms_list)
write("data/processed/good_pairs.traj", good_atoms_list)

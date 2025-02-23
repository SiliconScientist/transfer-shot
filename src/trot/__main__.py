import toml
import polars as pl


from trot.config import Config
from trot.processing import process_data
from trot.model import get_calculator, set_calculators

MODEL_NAMES = ["DimeNet++-S2EF-OC20-All", "SchNet-S2EF-OC20-All"]


def main():
    cfg = Config(**toml.load("config.toml"))
    adsorption_energies = {}
    energies, atoms_list = process_data(cfg)
    adsorption_energies["DFT"] = energies
    for name in MODEL_NAMES:
        calc = get_calculator(name)
        atoms_list = set_calculators(atoms_list, calc)
        energies = [atoms.get_potential_energy() for atoms in atoms_list]
        adsorption_energies[name] = energies
    df = pl.DataFrame(adsorption_energies)
    df.write_parquet(cfg.paths.processed.predictions)
    print(df)


if __name__ == "__main__":
    main()

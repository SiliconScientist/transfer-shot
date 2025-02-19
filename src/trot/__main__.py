import toml

from trot.config import Config
from trot.processing import get_atoms_list


def main():
    config = Config(**toml.load("config.toml"))

    # Print random seed
    atoms_list = get_atoms_list(config.paths.raw)
    print(len(atoms_list))


if __name__ == "__main__":
    main()

import toml

from trot.config import Config


def main():
    config = Config(**toml.load("config.toml"))

    # Print random seed
    print(f"Random seed: {config.random_seed}")


if __name__ == "__main__":
    main()

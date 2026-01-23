import json
import re
from pathlib import Path
import tomllib


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


cfg = load_config("config.toml")

subset_json_cfg = cfg.get("subset_json", {})
infile = subset_json_cfg.get("input", "")
outfile = subset_json_cfg.get("output", "")

with open(infile, "r") as f:
    data = json.load(f)
subset_data = {}
rxn_pattern = re.compile(r"->\s*H2O\*")
for rxn_key, entry in data.items():
    if not isinstance(entry, dict):
        continue
    keep = False
    if rxn_pattern.search(rxn_key):
        keep = True

    if keep:
        subset_data[rxn_key] = entry

with open(outfile, "w") as f:
    json.dump(subset_data, f, indent=2)

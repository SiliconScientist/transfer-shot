import json
import re

infile = "mace/raw_data/MamunHighT2019_adsorption.json"
outfile = "MamunHighT2019_h2o_adsorption.json"

with open(infile, "r") as f:
    data = json.load(f)

subset_data = {}

# Patterns:
# 1. Reaction key includes "-> H2O*"
rxn_pattern = re.compile(r"->\s*H2O\*")

for rxn_key, entry in data.items():
    # Skip any non-reaction metadata keys if you have them later
    if not isinstance(entry, dict):
        continue

    keep = False

    # Check reaction string
    if rxn_pattern.search(rxn_key):
        keep = True
    # else:
    #     # Check raw species names
    #     raw_block = entry.get("raw", {})
    #     if any(raw_pattern.search(spec_name) for spec_name in raw_block.keys()):
    #         keep = True

    if keep:
        subset_data[rxn_key] = entry

with open(outfile, "w") as f:
    json.dump(subset_data, f, indent=2)

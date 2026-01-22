Here's the data flow for the data processing:

For a Mamun subset:

Obtain the json file of the Mamun data "MamunHighT2019_adsorption.json (CatBench paper has this file).

Subset the Mamun dataset with the subset_json.py file.

Extract the atoms from the json subset with the extract_atoms_from_json.py script to get an extxyz file with the adsorption energies.

Relax these atoms with the uma_relax.py script, which retains the DFT reference labels in the extxyz format.

Check if any of the structures have exploded, or if theiradsorbates have migrated with the mismatch_finder.py script, which compares the DFT structures to the uma-relaxed structures.

Modify the config.tom file and obtain the MLIP predictions by running the src/trot/__main__.py script.
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

from ase.atoms import Atoms
from ase.io import read, write
import tomllib


def ase_atoms_from_atoms_json_str(atoms_json_str: str) -> Atoms:
    return read(StringIO(atoms_json_str), format="json")


def is_bound_component(raw_key: str) -> bool:
    """
    Keep only bound systems like 'H2Ostar', 'COstar', etc.
    Skip:
      - gas components like 'H2Ogas'
      - the clean slab 'star'
    """
    return raw_key.endswith("star") and raw_key != "star"


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
    # =====================
    # CONFIG (edit here)
    # =====================
    cfg = load_config("config.toml")
    extract_cfg = cfg.get("extract_atoms_from_json", {})

    input_json = Path(extract_cfg.get("input_json", ""))
    output_extxyz = Path(extract_cfg.get("output_extxyz", ""))

    # =====================
    # LOAD INPUT
    # =====================
    with input_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    frames: list[Atoms] = []
    n_failed = 0
    n_skipped = 0

    # =====================
    # PARSE STRUCTURES
    # =====================
    for reaction_key, reaction_obj in data.items():
        raw = reaction_obj.get("raw", {})
        if not isinstance(raw, dict):
            continue

        for raw_key, comp in raw.items():
            if not is_bound_component(raw_key):
                n_skipped += 1
                continue

            if not isinstance(comp, dict) or "atoms_json" not in comp:
                n_failed += 1
                print(
                    f"[WARN] Missing atoms_json for reaction='{reaction_key}' raw='{raw_key}'"
                )
                continue

            try:
                atoms = ase_atoms_from_atoms_json_str(comp["atoms_json"])

                # provenance metadata (goes into extxyz comment)
                atoms.info["reaction_key"] = reaction_key
                atoms.info["raw_name"] = raw_key  # e.g., H2Ostar
                atoms.info["adsorbate"] = raw_key[:-4]  # strip trailing 'star' -> 'H2O'

                frames.append(atoms)
            except Exception as e:
                n_failed += 1
                print(
                    f"[WARN] Failed to parse reaction='{reaction_key}' raw='{raw_key}': {e}"
                )

    # =====================
    # WRITE OUTPUT
    # =====================
    if not frames:
        raise SystemExit("No bound structures were found/parsed; nothing to write.")

    write(output_extxyz.as_posix(), frames, format="extxyz")
    print(
        f"Done. Wrote {len(frames)} bound frames to {output_extxyz} "
        f"(failed: {n_failed}, skipped non-bound: {n_skipped})."
    )


if __name__ == "__main__":
    main()

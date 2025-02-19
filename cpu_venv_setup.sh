#!/bin/bash
uv pip install torch==2.4.0
uv pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
uv pip install fairchem-core
uv pip install -e .
uv pip install -r requirements.txt
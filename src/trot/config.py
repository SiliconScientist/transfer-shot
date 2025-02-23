from pathlib import Path

from pydantic import BaseModel


class Raw(BaseModel):
    adsorbed: Path
    bare: Path


class Processed(BaseModel):
    predictions: Path


class Paths(BaseModel):
    raw: Raw
    processed: Processed


class Config(BaseModel):
    paths: Paths

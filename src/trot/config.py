from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    random_seed: int
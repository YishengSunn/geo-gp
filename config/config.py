import yaml
from pathlib import Path

class Config:
    def __init__(self, path: str | Path):
        with open(path, "r") as f:
            self._cfg = yaml.safe_load(f)

    def __getattr__(self, name):
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(name)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Settings:
    raw: dict[str, Any]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def strategy(self) -> dict[str, Any]:
        return self.raw["strategy"]

    @property
    def outputs(self) -> dict[str, Any]:
        return self.raw["outputs"]


def load_settings(path: str | Path = "config.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(raw=raw)

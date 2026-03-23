from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = {"data", "features", "model", "strategy", "outputs", "monitor"}


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

    @property
    def monitor(self) -> dict[str, Any]:
        return self.raw["monitor"]


def load_settings(path: str | Path = "config.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(raw))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Config file {config_path} is missing required sections: {missing_str}. "
            "Expected top-level keys: data, features, model, strategy, outputs, monitor."
        )

    return Settings(raw=raw)

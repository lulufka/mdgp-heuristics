import json
from pathlib import Path


def load_local_config() -> dict:
    path = Path("config.local.json")

    if not path.exists():
        raise FileNotFoundError(
            "config.local.json not found. Please create it with your local paths."
        )

    with open(path, "r") as f:
        return json.load(f)


_config = load_local_config()

KAPOCE_EXECUTABLE = Path(_config["kapoce_executable"])
KAPOCE_CONFIG = Path(_config["kapoce_config"])
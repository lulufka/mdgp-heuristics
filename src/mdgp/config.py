import json
from pathlib import Path
from typing import Any


def load_local_config() -> dict[str, Any]:
    """
    Loads local configuration variables from a JSON file.

    The file 'config.local.json' is expected to contain paths to external
    tools like the KaPoCE executable.

    Returns:
        dict[str, Any]: A dictionary containing configuration values.

    Raises:
        FileNotFoundError: If 'config.local.json' is not found in the working directory.
    """
    path = Path("config.local.json")

    if not path.exists():
        raise FileNotFoundError(
            "config.local.json not found. Please create it with your local paths."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_config = load_local_config()

KAPOCE_EXECUTABLE = Path(_config["kapoce_executable"])
KAPOCE_CONFIG = Path(_config["kapoce_config"])

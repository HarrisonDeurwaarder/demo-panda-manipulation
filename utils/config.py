import yaml
from typing import Any


config: dict[str, Any] | None = None


def load_config(path: str,) -> None:
    """
    Sets the program configuration

    Args:
        path (str): Config file path from project root
    """
    global config
    # Read the config
    with open(path) as f:
        config = yaml.safe_load(f)
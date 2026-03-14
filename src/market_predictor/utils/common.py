import os
import pickle
from typing import Any

import yaml


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if data is None:
        return {}
    return data


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_object(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_object(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)

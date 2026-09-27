from pathlib import Path

import yaml


def load_config(config_path=None):
    project_root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = project_root / "configs" / "experiment.yaml"

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for name, directory in config["datasets"].items():
        path = Path(directory)
        if not path.is_absolute():
            path = project_root / path
        config["datasets"][name] = str(path)

    return config

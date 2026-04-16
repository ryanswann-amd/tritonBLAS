import argparse
import os
import yaml


def parse_numeric(value):
    """Parse a string to a numeric type (int or float) if possible, otherwise return the original value."""
    if isinstance(value, str):
        try:
            return float(value) if "." in value or "e" in value else int(value)
        except ValueError:
            pass
    return value


def print_config(config, indent=0):
    """Print the configuration in a readable format."""
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(Config(value), indent=indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


class Config:
    """Configuration loader that supports YAML files and nested attribute access.

    Ported from GemmKernelSelection/Embedding/two_tower/config/config.py.
    Adapted for tritonBLAS kernel selection with Triton-specific config fields.
    """

    def __init__(self, config_path_or_dict=None):
        if config_path_or_dict is None:
            config_folder = os.path.dirname(os.path.abspath(__file__))
            config_path_or_dict = os.path.join(config_folder, "config.yaml")

        if isinstance(config_path_or_dict, str):
            assert os.path.isfile(
                config_path_or_dict
            ), f"Configuration file not found: {config_path_or_dict}."
            self._config_path = config_path_or_dict

            with open(config_path_or_dict, "r") as f:
                self._config = yaml.safe_load(f)

        elif isinstance(config_path_or_dict, dict):
            self._config = config_path_or_dict

        else:
            raise TypeError(
                "Config must be initialized with a file path or a dictionary."
            )

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"Attribute '{name}' not found")
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return parse_numeric(value)

        raise AttributeError(f"Attribute '{name}' not found")

    def __contains__(self, name):
        return name in self._config

    def __repr__(self):
        return f"Config({self._config})"

    def get(self, name, default=None):
        """Get a config value with a default fallback."""
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def to_dict(self):
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = parse_numeric(value)
        return result

    def keys(self):
        return self._config.keys()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration yaml file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args.config_path)
    print_config(config)

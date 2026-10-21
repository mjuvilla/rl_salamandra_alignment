"""Config files for distributed computing
    """
import importlib.resources as pkg_resources
import os
from typing import Literal

DistributedConfigs = Literal[
    "DSZero3Offload",  # Zero 3 with CPU offload
    "DSZero3",         # Zero 3 without offload
]


def get_distributed_config_path(
    config_name: DistributedConfigs
) -> str:
    """Get the absolute path to a config file for distributed computing

    Args:
        config_name (DistributedConfigs): Name of setup for distributed computing

    Returns:
        str: Path to config file
    """
    rl_salamanda_alignment_package_path = str(
        pkg_resources.files('rl_salamandra_alignment'))
    if config_name == "DSZero3Offload":
        json_config = "zero_3_mn5_config.json"
    elif config_name == "DSZero3":
        json_config = "zero_3_mn5_nooffload_config.json"
    else:
        raise ValueError(
            f"Unvalid distributed computing configuration: {config_name}")

    return os.path.join(rl_salamanda_alignment_package_path, "distributed_configs", json_config)

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import yaml


def label_loader(filename: str):
    """
    Load labels from a YAML file.
    Args:
        filename (str): Path to the YAML file containing labels.
    Returns:
        dict: A dictionary containing the labels.
    """
    with open(filename) as f:
        labels = yaml.safe_load(f)

    return labels

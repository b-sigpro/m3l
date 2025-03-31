# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import yaml


def label_loader(filename: str):
    with open(filename) as f:
        labels = yaml.safe_load(f)

    return labels

import yaml


def label_loader(filename: str):
    with open(filename) as f:
        labels = yaml.safe_load(f)

    return labels

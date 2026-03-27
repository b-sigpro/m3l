# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from torchvision.transforms.v2 import Transform


class FilterCrowdBBox(Transform):
    """Remove samples marked as ``iscrowd`` from detection targets.

    This transform filters out elements in detection targets (e.g.,
    ``boxes`` and ``labels``) based on the ``iscrowd`` flag.

    Args:
        targets (list[str] | None, optional): Keys to filter in addition to
            ``iscrowd``. If ``None``, defaults to ``["boxes", "labels"]``.

    Returns:
        dict: A dictionary with filtered targets, where all elements marked
        with ``iscrowd = True`` are removed.
    """

    def __init__(self, targets: list[str] | None = None):
        super().__init__()

        self.targets = ["iscrowd"] + (targets if targets is not None else ["boxes", "labels"])

    def transform(self, inpt, params):
        if not isinstance(inpt, dict):
            return inpt

        keep = ~inpt["iscrowd"]
        out = {k: v[keep] if k.lower() in self.targets else v for k, v in inpt.items()}
        assert all(len(out["iscrowd"]) == len(v) for v in out.values())

        return out

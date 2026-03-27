# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections import defaultdict
import json

import numpy as np

from torch.utils.data import Sampler, Subset


def randomized_cycle(iterator):
    listed_iterator = list(iterator)
    while True:
        yield from np.random.permutation(listed_iterator)


class BalancedSampler(Sampler[int]):
    """Sampler that balances sampling across label groups based on metadata.

    This sampler ensures that each label group, defined in the metadata file,
    is sampled fairly by cycling through randomized indices. It supports both
    full datasets and subsets.

    Args:
        dataset (Dataset | Subset): The dataset or a subset of it. If a Subset is given,
            its indices will be used to reference the original dataset.
        metadata_filename (str): Path to a JSON file containing label group metadata.
            The file should map each group name in ``dataset._dataset.dataset.grp_list``
            to a list of label identifiers.

    Returns:
        Iterator[int]: An infinite iterator over balanced, randomized indices.
    """

    def __init__(self, dataset, metadata_filename: str):
        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset = dataset.dataset
        else:
            indices = list(range(len(dataset)))

        self._length = len(indices)

        grp_list = dataset._dataset.dataset.grp_list

        with open(metadata_filename) as f:
            metadata = json.load(f)

        index_table = defaultdict(lambda: [])
        for index, orig_index in enumerate(indices):
            for ll in metadata[grp_list[orig_index]]:
                index_table[ll].append(index)

        self.index_generator = randomized_cycle([randomized_cycle(indices) for indices in index_table.values()])

    def __iter__(self):
        while True:
            yield next(next(self.index_generator))

    def __len__(self):
        return self._length

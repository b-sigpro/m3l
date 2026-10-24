# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import codecs
from functools import partial
import os

import torch
from torch import nn

from transformers import ClapTextModelWithProjection, RobertaTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LaionClapTextEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.tokenizer = RobertaTokenizerFast.from_pretrained("laion/clap-htsat-unfused")

        self.text_model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.lin = nn.Linear(512, d_model)

    def forward(self, labels):
        # text normalization
        flatten_labels = [item for sublist in labels for item in sublist]
        flatten_labels = map(partial(codecs.decode, encoding="unicode_escape"), flatten_labels)  # todo: unicode escape
        flatten_labels = map(str.lower, flatten_labels)  # lower case
        flatten_labels = map(lambda ll: ll.replace("_", " "), flatten_labels)  # replace _ with space
        flatten_labels = map(lambda ll: f"Sound of {ll}", flatten_labels)  # Add "Sound of" prefix
        flatten_labels = list(flatten_labels)

        # calculate embeddings
        with torch.no_grad():
            inputs = self.tokenizer(flatten_labels, return_tensors="pt", padding=True).to(self.text_model.device)
            emb = self.text_model(**inputs).text_embeds

        b = len(labels)
        n_labels = len(labels[0])
        return self.lin(emb.view(b, n_labels, 512))

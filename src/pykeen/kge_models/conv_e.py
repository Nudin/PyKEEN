# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

from typing import Dict, Optional

import torch
import torch.autograd
import torch.optim as optim
from pykeen.constants import (
    CONV_E_FEATURE_MAP_DROPOUT,
    CONV_E_HEIGHT,
    CONV_E_INPUT_CHANNELS,
    CONV_E_INPUT_DROPOUT,
    CONV_E_KERNEL_HEIGHT,
    CONV_E_KERNEL_WIDTH,
    CONV_E_NAME,
    CONV_E_OUTPUT_CHANNELS,
    CONV_E_OUTPUT_DROPOUT,
    CONV_E_WIDTH,
    EMBEDDING_DIM,
    GPU,
    LEARNING_RATE,
    MARGIN_LOSS,
    NUM_ENTITIES,
    NUM_RELATIONS,
    PREFERRED_DEVICE,
)
from pykeen.kge_models.base import BaseModule
from torch import nn
from torch.nn import Parameter, functional as F
from torch.nn.init import xavier_normal

__all__ = ["ConvE"]


class ConvE(BaseModule):
    """An implementation of ConvE [dettmers2017]_.

    .. [dettmers2017] Dettmers, T., *et al.* (2017) `Convolutional 2d knowledge graph embeddings
                      <https://arxiv.org/pdf/1707.01476.pdf>`_. arXiv preprint arXiv:1707.01476.

    .. seealso:: https://github.com/TimDettmers/ConvE/blob/master/model.py
    """

    model_name = CONV_E_NAME
    hyper_params = [
        EMBEDDING_DIM,
        CONV_E_INPUT_CHANNELS,
        CONV_E_OUTPUT_CHANNELS,
        CONV_E_HEIGHT,
        CONV_E_WIDTH,
        CONV_E_KERNEL_HEIGHT,
        CONV_E_KERNEL_WIDTH,
        CONV_E_INPUT_DROPOUT,
        CONV_E_FEATURE_MAP_DROPOUT,
        CONV_E_OUTPUT_DROPOUT,
        MARGIN_LOSS,
        LEARNING_RATE,
    ]

    def __init__(
        self,
        margin_loss: float,
        embedding_dim: int,
        ConvE_input_channels,
        ConvE_output_channels,
        ConvE_height,
        ConvE_width,
        ConvE_kernel_height,
        ConvE_kernel_width,
        conv_e_input_dropout,
        conv_e_output_dropout,
        conv_e_feature_map_dropout,
        random_seed: Optional[int] = None,
        preferred_device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        self.ConvE_height = ConvE_height
        self.ConvE_width = ConvE_width

        assert self.ConvE_height * self.ConvE_width == self.embedding_dim

        self.inp_drop = torch.nn.Dropout(conv_e_input_dropout)
        self.hidden_drop = torch.nn.Dropout(conv_e_output_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(conv_e_feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=ConvE_input_channels,
            out_channels=ConvE_output_channels,
            kernel_size=(ConvE_kernel_height, ConvE_kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )

        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(ConvE_input_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(ConvE_output_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        num_in_features = (
            ConvE_output_channels
            * (2 * self.ConvE_height - ConvE_kernel_height + 1)
            * (self.ConvE_width - ConvE_kernel_width + 1)
        )
        self.fc = torch.nn.Linear(num_in_features, self.embedding_dim)

        # Default optimizer for ConvE
        self.default_optimizer = optim.Adam

    def _init_embeddings(self):
        super()._init_embeddings()
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.register_parameter("b", Parameter(torch.zeros(self.num_entities)))

    def init(self):  # FIXME is this ever called?
        xavier_normal(self.entity_embeddings.weight.data)
        xavier_normal(self.relation_embeddings.weight.data)

    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print(
                "The model has not been fitted yet. Predictions are based on randomly initialized embeddings."
            )
            self._init_embeddings()

        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        batch_size = triples.shape[0]
        subject_batch = triples[:, 0:1]
        relation_batch = triples[:, 1:2]
        object_batch = triples[:, 2:3].view(-1)

        subject_batch_embedded = self.entity_embeddings(subject_batch).view(
            -1, 1, self.ConvE_height, self.ConvE_width
        )
        relation_batch_embedded = self.relation_embeddings(relation_batch).view(
            -1, 1, self.ConvE_height, self.ConvE_width
        )
        candidate_object_emebddings = self.entity_embeddings(object_batch)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([subject_batch_embedded, relation_batch_embedded], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # batch_size, num_output_channels * (2 * height - ConvE_kernel_height + 1) * (width - ConvE_kernel_width + 1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, candidate_object_emebddings.transpose(1, 0))
        scores = torch.sigmoid(x)

        max_scores, predictions = torch.max(scores, dim=1)

        # Class 0 represents false fact and class 1 represents true fact

        return predictions

    def forward(self, pos_batch, neg_batch):
        batch = torch.cat((pos_batch, neg_batch), dim=0)
        positive_labels = torch.ones(
            pos_batch.shape[0], dtype=torch.float, device=self.device
        )
        negative_labels = torch.zeros(
            neg_batch.shape[0], dtype=torch.float, device=self.device
        )
        labels = torch.cat([positive_labels, negative_labels], dim=0)

        perm = torch.randperm(labels.shape[0])

        batch = batch[perm]
        labels = labels[perm]

        batch_size = batch.shape[0]

        heads = batch[:, 0:1]
        relations = batch[:, 1:2]
        tails = batch[:, 2:3]

        # batch_size, num_input_channels, width, height
        heads_embs = self.entity_embeddings(heads).view(
            -1, 1, self.ConvE_height, self.ConvE_width
        )
        relation_embs = self.relation_embeddings(relations).view(
            -1, 1, self.ConvE_height, self.ConvE_width
        )
        tails_embs = self.entity_embeddings(tails).view(-1, self.embedding_dim)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([heads_embs, relation_embs], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # batch_size, num_output_channels * (2 * height - ConvE_kernel_height + 1) * (width - ConvE_kernel_width + 1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)

        scores = torch.sum(torch.mm(x, tails_embs.transpose(1, 0)), dim=1)

        predictions = torch.sigmoid(scores)
        loss = self.loss(predictions, labels)
        return loss

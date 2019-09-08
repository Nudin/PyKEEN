# -*- coding: utf-8 -*-

"""Implementation of UM."""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.autograd
from pykeen.constants import (
    NORM_FOR_NORMALIZATION_OF_ENTITIES,
    SCORING_FUNCTION_NORM,
    UM_NAME,
)
from pykeen.kge_models.base import BaseModule
from torch import nn

__all__ = ["UnstructuredModel"]

log = logging.getLogger(__name__)


class UnstructuredModel(BaseModule):
    """An implementation of Unstructured Model (UM) [bordes2014]_.

    .. [bordes2014] Bordes, A., *et al.* (2014). `A semantic matching energy function for learning with
                    multi-relational data <https://link.springer.com/content/pdf/10.1007%2Fs10994-013-5363-6.pdf>`_.
                    Machine
    """

    model_name = UM_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [
        SCORING_FUNCTION_NORM,
        NORM_FOR_NORMALIZATION_OF_ENTITIES,
    ]

    def __init__(
        self,
        margin_loss: float,
        embedding_dim: int,
        scoring_function: Optional[int] = 1,
        normalization_of_entities: Optional[int] = 2,
        random_seed: Optional[int] = None,
        preferred_device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)
        self.l_p_norm_entities = normalization_of_entities
        self.scoring_fct_norm = scoring_function

    def _init_embeddings(self):
        super()._init_embeddings()
        self._initialize()

    def _initialize(self):
        entity_embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-entity_embeddings_init_bound,
            b=entity_embeddings_init_bound,
        )

    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print(
                "The model has not been fitted yet. Predictions are based on randomly initialized embeddings."
            )
            self._init_embeddings()

        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        pos_scores = self._score_triples(batch_positives)
        neg_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(
            positive_scores=pos_scores, negative_scores=neg_scores
        )
        return loss

    def _score_triples(self, triples):
        return self._compute_scores(*self._get_triple_embeddings(triples))

    def _compute_scores(self, head_embeddings, tail_embeddings):
        # Add the vector element wise
        sum_res = head_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        distances = distances ** 2
        return distances

    def _get_triple_embeddings(self, triples):
        heads, tails = self.slice_triples(triples)
        return (self._get_entity_embeddings(heads), self._get_entity_embeddings(tails))

    @staticmethod
    def slice_triples(triples):
        return (triples[:, 0:1], triples[:, 2:3])

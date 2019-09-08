# -*- coding: utf-8 -*-

"""Implementation of RESCAL."""

from typing import Dict, Optional

import torch
import torch.autograd
from pykeen.constants import RESCAL_NAME, SCORING_FUNCTION_NORM
from pykeen.kge_models.base import BaseModule
from torch import nn

__all__ = ["RESCAL"]


class RESCAL(BaseModule):
    """An implementation of RESCAL [nickel2011]_.

    This model represents relations as matrices and models interactions between latent features.

    .. [nickel2011] Nickel, M., *et al.* (2011) `A Three-Way Model for Collective Learning on Multi-Relational Data
                    <http://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf>`_. ICML. Vol. 11.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """

    model_name = RESCAL_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM]

    def __init__(
        self,
        margin_loss: float,
        embedding_dim: int,
        scoring_function: Optional[int] = 1,
        random_seed: Optional[int] = None,
        preferred_device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        self.scoring_fct_norm = scoring_function

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(
            self.num_relations, self.embedding_dim * self.embedding_dim
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

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self._compute_loss(
            positive_scores=positive_scores, negative_scores=negative_scores
        )
        return loss

    def _score_triples(self, triples):
        return self._compute_scores(*self._get_triple_embeddings(triples))

    def _compute_scores(self, h_embs, r_embs, t_embs):
        # Compute score and transform result to 1D tensor
        m = r_embs.view(-1, self.embedding_dim, self.embedding_dim)
        h_embs = h_embs.unsqueeze(-1).permute([0, 2, 1])
        h_m_embs = torch.matmul(h_embs, m)
        t_embs = t_embs.unsqueeze(-1)
        scores = -torch.matmul(h_m_embs, t_embs).view(-1)

        # scores = torch.bmm(torch.transpose(h_emb, 1, 2), M)  # h^T M
        # scores = torch.bmm(scores, t_emb)  # (h^T M) h
        # scores = score.view(-1, 1)

        return scores

# -*- coding: utf-8 -*-

"""Implementations of various knowledge graph embedding models.

+---------------------------+---------------------------------------------------+
| Model Name                | Reference                                         |
|                           |                                                   |
+===========================+===================================================+
| TransE                    | :py:class:`pykeen.kge_models.TransE`              |
+---------------------------+---------------------------------------------------+
| TransH                    | :py:class:`pykeen.kge_models.TransH`              |
+---------------------------+---------------------------------------------------+
| TransR                    | :py:class:`pykeen.kge_models.TransR`              |
+---------------------------+---------------------------------------------------+
| TransD                    | :py:class:`pykeen.kge_models.TransD`              |
+---------------------------+---------------------------------------------------+
| ConvE                     | :py:class:`pykeen.kge_models.ConvE`               |
+---------------------------+---------------------------------------------------+
| Structured Embedding (SE) | :py:class:`pykeen.kge_models.StructuredEmbedding` |
+---------------------------+---------------------------------------------------+
| Unstructured Model (UM)   | :py:class:`pykeen.kge_models.UnstructuredModel`   |
+---------------------------+---------------------------------------------------+
| RESCAL                    | :py:class:`pykeen.kge_models.RESCAL`              |
+---------------------------+---------------------------------------------------+
| ERMLP                     | :py:class:`pykeen.kge_models.ERMLP`               |
+---------------------------+---------------------------------------------------+
| DistMult                  | :py:class:`pykeen.kge_models.DistMult`            |
+---------------------------+---------------------------------------------------+
"""

from typing import Dict

from pykeen.constants import KG_EMBEDDING_MODEL_NAME
from pykeen.kge_models.conv_e import ConvE  # noqa: F401
from pykeen.kge_models.distmult import DistMult  # noqa: F401
from pykeen.kge_models.ermlp import ERMLP  # noqa: F401
from pykeen.kge_models.rescal import RESCAL  # noqa: F401
from pykeen.kge_models.structured_embedding import StructuredEmbedding  # noqa: F401
from pykeen.kge_models.trans_d import TransD  # noqa: F401
from pykeen.kge_models.trans_e import TransE  # noqa: F401
from pykeen.kge_models.trans_h import TransH  # noqa: F401
from pykeen.kge_models.trans_r import TransR  # noqa: F401
from pykeen.kge_models.unstructured_model import UnstructuredModel  # noqa: F401
from torch.nn import Module

__all__ = [
    "TransE",
    "TransH",
    "TransR",
    "TransD",
    "ConvE",
    "StructuredEmbedding",
    "UnstructuredModel",
    "RESCAL",
    "ERMLP",
    "DistMult",
    "get_kge_model",
    "KGE_MODELS",
    "get_kge_model",
]

#: A mapping from KGE model names to KGE model classes
KGE_MODELS = {
    TransE.model_name: TransE,
    TransH.model_name: TransH,
    TransD.model_name: TransD,
    TransR.model_name: TransR,
    StructuredEmbedding.model_name: StructuredEmbedding,
    UnstructuredModel.model_name: UnstructuredModel,
    DistMult.model_name: DistMult,
    ERMLP.model_name: ERMLP,
    RESCAL.model_name: RESCAL,
    ConvE.model_name: ConvE,
}


def get_kge_model(config: Dict) -> Module:
    """Get an instance of a knowledge graph embedding model with the given configuration."""
    kge_model_name = config[KG_EMBEDDING_MODEL_NAME]
    kge_model_cls = KGE_MODELS.get(kge_model_name)

    if kge_model_cls is None:
        raise ValueError(f"Invalid KGE model name: {kge_model_name}")

    return kge_model_cls(**config)

# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from typing import Any, List, Mapping, Optional, Union, Iterable, Tuple

import logging
import timeit
from tqdm import trange
import random

import numpy as np
import torch
import rdflib
from torch import nn
import torch.optim as optim
from pykeen.utilities.triples_creation_utils import create_mappings, create_mapped_triples
import pykeen.constants as pkc

from pykeen.constants import (
    EMBEDDING_DIM, GPU, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE,
)

__all__ = [
    'BaseModule',
    'slice_triples',
]

log = logging.getLogger(__name__)

class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self,
                 margin_loss: float,
                 embedding_dim: int,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 ) -> None:
        super().__init__()

        # Device selection
        self._get_device(preferred_device)

        self.random_seed = random_seed

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

        # Loss
        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            reduction='mean' if self.margin_ranking_loss_size_average else 'sum'
        )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = None
        #: The number of unique relation types in the knowledge graph
        self.num_relations = None
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        # Default optimizer for all classes
        self.default_optimizer = optim.SGD

        # Instance attributes that are defined when calling other functions
        # Calling data load function
        self.entity_label_to_id = None
        self.relation_label_to_id = None

        # Calling fit function
        self.entity_embeddings = None
        self.learning_rate = None
        self.num_epochs = None
        self.batch_size = None

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _get_device(self,
                    device: str = 'cpu',
                    ) -> None:
        """Get the Torch device to use."""
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                print('No cuda devices were available. The model runs on CPU')
        else:
            self.device = torch.device('cpu')

    def _to_cpu(self):
        """Transfer the entire model to CPU"""
        self._get_device('cpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _to_gpu(self):
        """Transfer the entire model to GPU"""
        self._get_device('gpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        y = torch.FloatTensor([-1])
        y = y.expand(positive_scores.shape[0]).to(self.device)
        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def load_triples_from_path(self,
                     data_paths: Union[str, List[str]]):
        """
        Loads triples from files given their paths, creates mappings and returns the mapped triples
        :param data_paths: The paths for all files that are going to be used for training and testing
        :return: List where each items represents the mapped triples of a file
        """

        if isinstance(data_paths, str):
            triples = load_data(data_paths)
            mapped_triples = self.load_triples(triples)
        else:
            triples_list = [load_data(data_path) for data_path in data_paths]
            mapped_triples = self.load_triples(triples_list)

        return mapped_triples

    def load_triples(self,
                     triples_in: Union[np.ndarray, List[np.ndarray]]):
        """
        Loads triples from arrays, creates mappings and returns the mapped triples
        :param data_paths: The paths for all files that are going to be used for training and testing
        :return: List where each items represents the mapped triples of a file
        """

        if isinstance(triples_in, np.ndarray):
            all_triples = triples_in
            self.entity_label_to_id, self.relation_label_to_id = create_mappings(triples=all_triples)
            mapped_triples, _, _ = create_mapped_triples(
                triples=all_triples,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=self.relation_label_to_id
            )
        else:
            all_triples: np.ndarray = np.concatenate(triples_in, axis=0)
            self.entity_label_to_id, self.relation_label_to_id = create_mappings(triples=all_triples)
            mapped_triples = [create_mapped_triples(triples,
                                                    entity_label_to_id=self.entity_label_to_id,
                                                    relation_label_to_id=self.relation_label_to_id)[0]
                              for triples in triples_in]

        self.num_entities = len(self.entity_label_to_id)
        self.num_relations = len(self.relation_label_to_id)

        return mapped_triples

    def map_triples_from_path(self,
                    data_paths: Union[str, List[str]]):
        """
        Loads triples and returns the mapped triples given the mappings of the model
        :param data_paths: The paths for the triples files that should be mapped
        :return: List where each items represents the mapped triples of a file
        """

        if isinstance(data_paths, str):
            triples = load_data(data_paths)
            mapped_triples = self.map_triples(triples)
        else:
            triples_list = [load_data(data_path) for data_path in data_paths]
            mapped_triples = self.map_triples(triples_list)

        return mapped_triples

    def map_triples(self,
                    triples_in: Union[np.ndarray, List[np.ndarray]]):
        """
        Loads triples and returns the mapped triples given the mappings of the model
        :param data_paths: The paths for the triples files that should be mapped
        :return: List where each items represents the mapped triples of a file
        """

        if isinstance(triples_in, np.ndarray):
            mapped_triples, _, _ = create_mapped_triples(
                triples=triples_in,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=self.relation_label_to_id
            )
        else:
            mapped_triples = [create_mapped_triples(triples=triples,
                                                    entity_label_to_id=self.entity_label_to_id,
                                                    relation_label_to_id=self.relation_label_to_id)[0]
                              for triples in triples_in]

        return mapped_triples


    def _init_embeddings(self):
        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            max_norm=self.entity_embedding_max_norm,
            norm_type=self.entity_embedding_norm_type,
        )

    def predict_objects(self,
                       subject: str,
                       relation) -> str:
        """"""
        subject_id = self.entity_label_to_id[subject]
        relation_id = self.relation_label_to_id[relation]
        object_ids = np.array(list(self.entity_label_to_id.values()))
        object_values = np.array(list(self.entity_label_to_id.keys()))
        # Filter the subject out of the entity
        indexing = object_ids == subject_id
        object_ids = object_ids[~(indexing)]
        object_values = object_values[~(indexing)]
        subject_ids = np.full_like(object_ids, subject_id)
        relation_ids = np.full_like(object_ids, relation_id)
        triples = np.vstack((subject_ids, relation_ids, object_ids)).T

        scores = self.predict(torch.tensor(triples, dtype=torch.long, device=self.device))

        sorting = scores.argsort()

        scored_objects = np.vstack((object_values[sorting[::-1]], scores[sorting[::-1]]))

        return scored_objects.T

    def predict_best_object(self,
                       subject: str,
                       relation) -> str:
        """"""
        return self.predict_objects(subject, relation)[0]

    def predict_subjects(self,
                       obj: str,
                       relation: str) -> str:
        """"""
        object_id = self.entity_label_to_id[obj]
        relation_id = self.relation_label_to_id[relation]
        subject_ids = np.array(list(self.entity_label_to_id.values()))
        subject_values = np.array(list(self.entity_label_to_id.keys()))
        # Filter the subject out of the entity
        indexing = subject_ids == object_id
        subject_ids = subject_ids[~(indexing)]
        subject_values = subject_values[~(indexing)]
        object_ids = np.full_like(subject_ids, object_id)
        relation_ids = np.full_like(subject_ids, relation_id)
        triples = np.vstack((subject_ids, relation_ids, object_ids)).T

        scores = self.predict(torch.tensor(triples, dtype=torch.long, device=self.device))

        sorting = scores.argsort()

        scored_subjects = np.vstack((subject_values[sorting], scores[sorting]))

        return scored_subjects.T

    def predict_best_subject(self,
                       obj: str,
                       relation) -> str:
        """"""
        return self.predict_subjects(obj, relation)[0]

    def fit(self,
            pos_triples: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            batch_size: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            weight_decay: Optional[float] = 0,
            tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """
        Trains the kge model with the given parameters
        :param pos_triples: Positive triples to train on
        :param learning_rate: Learning rate for the optimizer
        :param num_epochs: Number of epochs to train
        :param batch_size: Batch size for training
        :param optimizer: Pytorch optimizer class to use for training
        :param tqdm_kwargs: Keyword arguments that should be used for the tdqm.trange class
        :return: loss_per_epoch: The loss of each epoch during training
        """
        self._init_embeddings()

        self.to(self.device)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        if optimizer is None:
            # Initialize the standard optimizer with the correct parameters
            self.optimizer = self.default_optimizer(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            # Initialize the optimizer given as attribute
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

        log.info(f'****Run Model On {str(self.device).upper()}****')

        loss_per_epoch = []
        num_pos_triples = pos_triples.shape[0]
        all_entities = np.arange(self.num_entities)

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')
        if tqdm_kwargs:
            _tqdm_kwargs.update(tqdm_kwargs)

        for epoch in trange(self.num_epochs, **_tqdm_kwargs):
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            num_positives = self.batch_size
            pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=num_positives)
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                # TODO: Implement helper functions for different negative sampling approaches
                current_batch_size = len(pos_batch)

                batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)

                # num_subj_corrupt = len(pos_batch) // 2
                # num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.device)

                # corrupted_subj_indices = np.random.choice(np.arange(0, self.num_entities), size=num_subj_corrupt)
                # corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
                # subject_based_corrupted_triples = np.concatenate(
                #     [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)
                #
                # corrupted_obj_indices = np.random.choice(np.arange(0, self.num_entities), size=num_obj_corrupt)
                # corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))
                #
                # object_based_corrupted_triples = np.concatenate(
                #     [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)
                #
                # neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)
                #
                # neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=self.device)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                self.optimizer.zero_grad()
                loss = self(pos_batch, None)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                self.optimizer.step()

            # log.info(f"Epoch {str(epoch)} took {str(round(stop - start))} seconds \n")
            # Track epoch loss
            loss_per_epoch.append(current_epoch_loss / len(pos_triples))

        stop_training = timeit.default_timer()
        log.info(f"Training took {str(round(stop_training - start_training))} seconds \n")

        return loss_per_epoch


def slice_triples(triples: np.ndarray):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t

def _split_list_in_batches(input_list: np.ndarray,
                           batch_size: int):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def load_data(path: Union[str, Iterable[str]]) -> np.ndarray:
    """Load data given the *path*."""
    if isinstance(path, str):
        return _load_data_helper(path)

    return np.concatenate([
        _load_data_helper(p)
        for p in path
    ])

def _load_data_helper(path: str) -> np.ndarray:
    for prefix, handler in pkc.IMPORTERS.items():
        if path.startswith(f'{prefix}:'):
            return handler(path[len(f'{prefix}:'):])

    if path.endswith('.tsv'):
        return np.reshape(np.loadtxt(
            fname=path,
            dtype=str,
            comments='@Comment@ Subject Predicate Object',
            delimiter='\t',
        ), newshape=(-1, 3))

    if path.endswith('.nt'):
        g = rdflib.Graph()
        g.parse(path, format='nt')
        return np.array(
            [
                [str(s), str(p), str(o)]
                for s, p, o in g
            ],
            dtype=np.str,
        )

    raise ValueError('''The argument to _load_data must be one of the following:

    - A string path to a .tsv file containing 3 columns corresponding to subject, predicate, and object
    - A string path to a .nt RDF file serialized in N-Triples format
    - A string NDEx network UUID prefixed by "ndex:" like in ndex:f93f402c-86d4-11e7-a10d-0ac135e8bacf
    ''')

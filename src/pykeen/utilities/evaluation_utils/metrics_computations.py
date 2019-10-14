# -*- coding: utf-8 -*-

"""Script to compute mean rank and hits@k."""

import logging
import timeit
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ...constants import EMOJI

log = logging.getLogger(__name__)

DEFAULT_HITS_AT_K = [1, 3, 5, 10]


def update_hits_at_k(
    hits_at_k_values: Dict[int, List[float]],
    rank_of_positive_subject_based: int,
    rank_of_positive_object_based: int,
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        if rank_of_positive_subject_based <= k:
            values.append(1.0)
        else:
            values.append(0.0)

        if rank_of_positive_object_based <= k:
            values.append(1.0)
        else:
            values.append(0.0)


def _create_corrupted_triples(triple, all_entities, device):
    candidate_entities_subject_based = all_entities[
        all_entities != triple[0:1]
    ].reshape((-1, 1))
    candidate_entities_object_based = all_entities[all_entities != triple[2:3]].reshape(
        (-1, 1)
    )

    # Extract current test tuple: Either (subject,predicate) or (predicate,object)
    tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
    tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

    # Copy current test tuple
    tuples_subject_based = np.repeat(
        a=tuple_subject_based, repeats=candidate_entities_subject_based.shape[0], axis=0
    )
    tuples_object_based = np.repeat(
        a=tuple_object_based, repeats=candidate_entities_object_based.shape[0], axis=0
    )

    corrupted_subject_based = np.concatenate(
        [candidate_entities_subject_based, tuples_subject_based], axis=1
    )

    corrupted_object_based = np.concatenate(
        [tuples_object_based, candidate_entities_object_based], axis=1
    )

    return corrupted_subject_based, corrupted_object_based


def _filter_corrupted_triples(
    corrupted_subject_based, corrupted_object_based, all_pos_triples
):
    s = corrupted_object_based[0, 0].item()
    p = corrupted_object_based[0, 1].item()
    o = corrupted_object_based[0, 2].item()
    pos_subj = all_pos_triples[
        (all_pos_triples[:, 1] == p) & (all_pos_triples[:, 2] == o), 0
    ]
    mask = np.in1d(corrupted_subject_based[:, 0], pos_subj, invert=True)
    corrupted_subject_based = corrupted_subject_based[mask]

    pos_obj = all_pos_triples[
        (all_pos_triples[:, 1] == p) & (all_pos_triples[:, 0] == s), 2
    ]
    mask = np.in1d(corrupted_object_based[:, 2], pos_obj, invert=True)
    corrupted_object_based = corrupted_object_based[mask]

    if len(corrupted_object_based) + len(corrupted_subject_based) == 0:
        raise Exception(
            "User selected filtered metric computation, but all corrupted triples exists"
            "also as positive triples."
        )

    return corrupted_subject_based, corrupted_object_based


def _compute_filtered_rank(
    kg_embedding_model,
    pos_triple,
    corrupted_subject_based,
    corrupted_object_based,
    device,
    all_pos_triples,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples:
    """
    corrupted_subject_based, corrupted_object_based = _filter_corrupted_triples(
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        all_pos_triples=all_pos_triples,
    )

    return _compute_rank(
        kg_embedding_model=kg_embedding_model,
        pos_triple=pos_triple,
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        device=device,
        all_pos_triples=all_pos_triples,
    )


def _compute_rank(
    kg_embedding_model,
    pos_triple,
    corrupted_subject_based,
    corrupted_object_based,
    device,
    all_pos_triples=None,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples: This parameter isn't used but is necessary for compatability
    """
    corrupted_subject_based = torch.tensor(
        corrupted_subject_based, dtype=torch.long, device=device
    )
    corrupted_object_based = torch.tensor(
        corrupted_object_based, dtype=torch.long, device=device
    )
    scores_of_corrupted_subjects = kg_embedding_model.predict(corrupted_subject_based)
    scores_of_corrupted_objects = kg_embedding_model.predict(corrupted_object_based)

    score_of_positive = kg_embedding_model.predict(
        torch.tensor([pos_triple], dtype=torch.long, device=device)
    )

    rank_of_positive_subject_based = (
        scores_of_corrupted_subjects.shape[0]
        - np.greater(scores_of_corrupted_subjects, score_of_positive).sum()
    )

    rank_of_positive_object_based = (
        scores_of_corrupted_objects.shape[0]
        - np.greater(scores_of_corrupted_objects, score_of_positive).sum()
    )

    return (rank_of_positive_subject_based + 1, rank_of_positive_object_based + 1)


@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


def compute_metric_results(
    kg_embedding_model,
    mapped_train_triples,
    mapped_test_triples,
    device,
    filter_neg_triples=False,
    ks: Optional[List[int]] = None,
    *,
    use_tqdm: bool = True,
) -> MetricResults:
    """Compute the metric results.

    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_test_triples:
    :param device:
    :param filter_neg_triples:
    :param ks:
    :param use_tqdm: Should a progress bar be shown?
    :return:
    """
    start = timeit.default_timer()

    ranks: List[int] = []
    hits_at_k_values = {k: [] for k in (ks or DEFAULT_HITS_AT_K)}
    kg_embedding_model = kg_embedding_model.eval()
    kg_embedding_model = kg_embedding_model.to(device)

    all_entities = np.arange(kg_embedding_model.num_entities)

    all_pos_triples = np.concatenate(
        [mapped_train_triples, mapped_test_triples], axis=0
    )

    compute_rank_fct: Callable[..., Tuple[int, int]] = (
        _compute_filtered_rank if filter_neg_triples else _compute_rank
    )

    if use_tqdm:
        mapped_test_triples = tqdm(
            mapped_test_triples, desc=f"{EMOJI} corrupting triples"
        )
    for pos_triple in mapped_test_triples:
        corrupted_subject_based, corrupted_object_based = _create_corrupted_triples(
            triple=pos_triple, all_entities=all_entities, device=device
        )

        rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
            kg_embedding_model=kg_embedding_model,
            pos_triple=pos_triple,
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            device=device,
            all_pos_triples=all_pos_triples,
        )

        ranks.append(rank_of_positive_subject_based)
        ranks.append(rank_of_positive_object_based)

        # Compute hits@k for k in {1,3,5,10}
        update_hits_at_k(
            hits_at_k_values,
            rank_of_positive_subject_based=rank_of_positive_subject_based,
            rank_of_positive_object_based=rank_of_positive_object_based,
        )

    mean_rank = float(np.mean(ranks))
    hits_at_k: Dict[int, float] = {
        k: np.mean(values) for k, values in hits_at_k_values.items()
    }

    stop = timeit.default_timer()
    log.info("Evaluation took %.2fs seconds", stop - start)

    return MetricResults(mean_rank=mean_rank, hits_at_k=hits_at_k)

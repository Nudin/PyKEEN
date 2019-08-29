from enum import Enum

import numpy as np
from pykeen.kge_models.utils import slice_triples


class SamplingStrategy(Enum):
    RANDOM = "RANDOM"
    CORRUPTION = "CORRUPTION"
    TYPED = "TYPED"
    NEARMISS = "NEARMISS"
    NEXTNEIGHBOR = "NEXTNEIGHBOR"


def negative_sample(strategy, num, pos_batch, num_entities, all_entities):
    if strategy == SamplingStrategy.CORRUPTION:
        return ns_corruption(num, pos_batch, num_entities, all_entities)
    raise NotImplementedError(
        f"Negative sampling strategy {strategy} is not implemented"
    )


def ns_corruption(num, pos_batch, num_entities, all_entities):
    batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
    num_subj_corrupt = num // 2
    num_obj_corrupt = num - num_subj_corrupt

    corrupted_subj_indices = np.random.choice(
        np.arange(0, num_entities), size=num_subj_corrupt
    )
    corrupted_subjects = np.reshape(
        all_entities[corrupted_subj_indices], newshape=(-1, 1)
    )
    subject_based_corrupted_triples = np.concatenate(
        [
            corrupted_subjects,
            batch_relations[:num_subj_corrupt],
            batch_objs[:num_subj_corrupt],
        ],
        axis=1,
    )

    corrupted_obj_indices = np.random.choice(
        np.arange(0, num_entities), size=num_obj_corrupt
    )
    corrupted_objects = np.reshape(
        all_entities[corrupted_obj_indices], newshape=(-1, 1)
    )
    object_based_corrupted_triples = np.concatenate(
        [
            batch_subjs[num_subj_corrupt:],
            batch_relations[num_subj_corrupt:],
            corrupted_objects,
        ],
        axis=1,
    )

    return np.concatenate(
        [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
    )

from enum import Enum

import numpy as np
from pykeen.kge_models.utils import slice_triples


class SamplingStrategy(Enum):
    RANDOM = "RANDOM"
    CORRUPTION = "CORRUPTION"
    TYPED = "TYPED"
    NEARMISS = "NEARMISS"
    NEXTNEIGHBOR = "NEXTNEIGHBOR"


def negative_sample(strategy, num, pos_batch, num_entities):
    """
    In these implementations we assumes that all entities are continuously
    numbered starting with 0.
    """
    if strategy == SamplingStrategy.CORRUPTION:
        return random_corruption(num, pos_batch, num_entities)
    raise NotImplementedError(
        f"Negative sampling strategy {strategy} is not implemented"
    )


def random_corruption(num, pos_batch, num_entities):
    batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
    num_subj_corrupt = num // 2
    num_obj_corrupt = num - num_subj_corrupt

    corrupted_subjects = np.random.choice(num_entities, size=num_subj_corrupt)
    corrupted_subjects = np.reshape(corrupted_subjects, newshape=(-1, 1))
    subject_based_corrupted_triples = np.concatenate(
        [
            corrupted_subjects,
            batch_relations[:num_subj_corrupt],
            batch_objs[:num_subj_corrupt],
        ],
        axis=1,
    )

    corrupted_objects = np.random.choice(num_entities, size=num_obj_corrupt)
    corrupted_objects = np.reshape(corrupted_objects, newshape=(-1, 1))
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


def typed_corruption(num, pos_batch, subjects_by_relation, objects_by_relation):
    """

    subjects_by_relation: dict{relation, list(entities)}
    objects_by_relation: dict{relation, list(entities)}
    """
    batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
    num_subj_corrupt = num // 2
    num_obj_corrupt = num - num_subj_corrupt

    subject_based_corrupted_triples = []
    for i in num_subj_corrupt:
        relation = batch_relations[i]
        subject = np.random.choise(subjects_by_relation[relation])
        subject_based_corrupted_triples.append([subject, relation, batch_objs[i]])

    object_based_corrupted_triples = []
    for i in num_obj_corrupt:
        relation = batch_relations[i]
        obj = np.random.choise(objects_by_relation[relation])
        object_based_corrupted_triples.append([batch_subjs[i], relation, obj])

    return np.concatenate(
        [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
    )


def weighted_corruption(num, pos_batch, subjects_by_relation, objects_by_relation):
    """

    subjects_by_relation: dict{relation, [list(entities), list(weigths)]}
    objects_by_relation: dict{relation, [list(entities), list (weigths)]}
    """
    batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
    num_subj_corrupt = num // 2
    num_obj_corrupt = num - num_subj_corrupt

    subject_based_corrupted_triples = []
    for i in num_subj_corrupt:
        relation = batch_relations[i]
        subject = np.random.choise(
            subjects_by_relation[relation][0], subjects_by_relation[relation][0]
        )
        subject_based_corrupted_triples.append([subject, relation, batch_objs[i]])

    object_based_corrupted_triples = []
    for i in num_obj_corrupt:
        relation = batch_relations[i]
        obj = np.random.choise(
            objects_by_relation[relation][0], objects_by_relation[relation][0]
        )
        object_based_corrupted_triples.append([batch_subjs[i], relation, obj])

    return np.concatenate(
        [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
    )


def filter_entities(pos_triples):
    relations = pos_triples[:, 1:2]
    subjects_by_relation = {}
    objects_by_relation = {}
    for relation in np.unique(relations):
        triples = pos_triples[pos_triples[1] == relation]
        subjects_by_relation.append(triples[0])
        objects_by_relation.append(triples[2])
    # TODO: move loop into numpy for better performance, possibly by:
    # np.split(a[:, 0], np.cumsum(np.unique(a[:, 1], return_counts=True)[1])[:-1])
    return subjects_by_relation, objects_by_relation

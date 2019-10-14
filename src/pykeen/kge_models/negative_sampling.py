from enum import Enum
from itertools import repeat

import numpy as np
from pykeen.constants import CORRUPTION, RANDOM, TYPEDCORRUPTION, WEIGHTED
from pykeen.kge_models.utils import slice_triples


class NegativeSampler:
    """
    In these implementations we assumes that all entities are continuously
    numbered starting with 0.
    """

    def __init__(self, strategy, pos_triples, num_entities, num_relations):
        """
        Bind the chosen sampling strategy to .sample() and save the positive
        triples and the upper range of the entities
        """
        self.num_entities = num_entities
        self.pos_triples = pos_triples
        if strategy == CORRUPTION:
            self.sample = self.random_corruption
        elif strategy == TYPEDCORRUPTION:
            self.calc_entity_weights(num_relations)
            self.sample = self.typed_corruption
        elif strategy == WEIGHTED:
            self.calc_entity_weights(num_relations)
            self.sample = self.weighted_corruption
        else:
            raise NotImplementedError(
                f"Negative sampling strategy {strategy} is not implemented"
            )

    def random_corruption(self, pos_batch, mult=1):
        """
        Replace object or subject of positive triples with random entities.

        mult: number of negative samples per positive sample
        """
        batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
        if mult > 1:
            batch_subjs = np.tile(batch_subjs, mult)
            batch_relations = np.tile(batch_relations, mult)
            batch_objs = np.tile(batch_objs, mult)
        num = len(pos_batch) * mult
        num_subj_corrupt = num // 2
        num_obj_corrupt = num - num_subj_corrupt

        corrupted_subjects = np.random.choice(self.num_entities, size=num_subj_corrupt)
        subject_based_corrupted_triples = np.column_stack(
            (
                corrupted_subjects,
                batch_relations[:num_subj_corrupt],
                batch_objs[:num_subj_corrupt],
            )
        )

        corrupted_objects = np.random.choice(self.num_entities, size=num_obj_corrupt)
        object_based_corrupted_triples = np.column_stack(
            (
                batch_subjs[num_subj_corrupt:],
                batch_relations[num_subj_corrupt:],
                corrupted_objects,
            )
        )

        return np.concatenate(
            [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
        )

    def typed_corruption(self, pos_batch, mult=1):
        """
        Replace object or subject of positive triples with random entities
        that are also used as subject/object for the used relation at some
        positive sample.
        """
        batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
        num_pos = len(pos_batch)
        num = num_pos * mult
        num_subj_corrupt = num // 2

        corrupted_subjects = np.array(
            list(
                map(
                    np.random.choice,
                    self.subjects_by_relation[batch_relations[:num_subj_corrupt]][:, 0],
                )
            )
        )
        subject_based_corrupted_triples = np.column_stack(
            (
                corrupted_subjects,
                batch_relations[:num_subj_corrupt],
                batch_objs[:num_subj_corrupt],
            )
        )
        corrupted_object = np.array(
            list(
                map(
                    np.random.choice,
                    self.objects_by_relation[batch_relations[num_subj_corrupt:]][:, 0],
                )
            )
        )
        object_based_corrupted_triples = np.column_stack(
            (
                batch_subjs[num_subj_corrupt:],
                batch_relations[num_subj_corrupt:],
                corrupted_object,
            )
        )

        return np.concatenate(
            [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
        )

    def weighted_corruption(self, pos_batch, mult=1):
        """
        Replace object or subject of positive triples with random entities
        that are also used as subject/object for the used relation at some
        positive sample.
        The chance of an entity getting chosen are the bigger the more often
        they are found in positive triples.
        """
        batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)
        num_pos = len(pos_batch)
        num = num_pos * mult
        num_subj_corrupt = num // 2

        corrupted_subjects = np.array(
            list(
                map(
                    np.random.choice,
                    self.subjects_by_relation[batch_relations[:num_subj_corrupt]][:, 0],
                    repeat(1),
                    repeat(True),
                    self.subjects_by_relation[batch_relations][:, 1],
                )
            )
        )
        subject_based_corrupted_triples = np.column_stack(
            (
                corrupted_subjects,
                batch_relations[:num_subj_corrupt],
                batch_objs[:num_subj_corrupt],
            )
        )

        corrupted_object = np.array(
            list(
                map(
                    np.random.choice,
                    self.objects_by_relation[batch_relations[num_subj_corrupt:]][:, 0],
                    repeat(1),
                    repeat(True),
                    self.objects_by_relation[batch_relations][:, 1],
                )
            )
        )
        object_based_corrupted_triples = np.column_stack(
            (
                batch_subjs[num_subj_corrupt:],
                batch_relations[num_subj_corrupt:],
                corrupted_object,
            )
        )

        return np.concatenate(
            [subject_based_corrupted_triples, object_based_corrupted_triples], axis=0
        )

    def calc_entity_weights(self, num_relations):
        """
        Counts which entities are used how often with the different relations,
        stores the information in:
        self.subjects_by_relation: dict{relation, [list(entities), list(weigths)]}
        self.objects_by_relation: dict{relation, [list(entities), list (weigths)]}
        """
        subjects_by_relation = []
        objects_by_relation = []
        for relation in range(num_relations):
            triples = self.pos_triples[self.pos_triples[:, 1] == relation]
            entities, weights = np.unique(triples[:, 0], return_counts=True)
            subjects_by_relation.append([entities, weights / weights.sum()])
            entities, weights = np.unique(triples[:, 2], return_counts=True)
            objects_by_relation.append([entities, weights / weights.sum()])
        # TODO: move loop into numpy for better performance, possibly by:
        # np.split(a[:, 0], np.cumsum(np.unique(a[:, 1], return_counts=True)[1])[:-1])
        self.subjects_by_relation = np.array(list(subjects_by_relation.values()))
        self.objects_by_relation = np.array(list(objects_by_relation.values()))

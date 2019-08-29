# -*- coding: utf-8 -*-

"""Utilities used by the KGE models."""

import numpy as np


def slice_triples(triples: np.ndarray):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t

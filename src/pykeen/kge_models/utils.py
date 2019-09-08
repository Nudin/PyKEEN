# -*- coding: utf-8 -*-

"""Utilities used by the KGE models."""

import numpy as np


def slice_triples(triples: np.ndarray):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    return h, r, t

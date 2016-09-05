#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An example of "check_random_state", which is a utility function
in a scikit-learn package
"""

from __future__ import print_function
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

class RandomStateSample(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X=None, y=None, random_state=None):
        if random_state is None:
            random_state = self.random_state
        self._rng = check_random_state(random_state)
        return self._rng.randn(10)

m = RandomStateSample(random_state=1234)
print("seed = 1234 @ init\n", m.fit())

m = RandomStateSample()
print("seed = 1234 @ fit\n", m.fit(random_state=1234))

m = RandomStateSample()
print("seed = None\n", m.fit())
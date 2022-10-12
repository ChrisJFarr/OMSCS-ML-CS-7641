"""
Solver algorithms

    randomized hill climbing
    simulated annealing
    a genetic algorithm
    MIMIC

"""

import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose import (
    random_hill_climb, 
    simulated_annealing, 
    genetic_alg,
    mimic
)
import numpy as np

# Might only need a model class for simulated annealing and neural network


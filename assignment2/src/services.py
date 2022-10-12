import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import hydra
from src.evaluation import Assignment2Evaluation
import numpy as np


def run(config):
    # fitness_inputs = dict()

    fitness = hydra.utils.instantiate(
        config.experiments.dataset.fitness
        )
    problem = hydra.utils.instantiate(
        config.experiments.dataset.problem, fitness_fn=fitness)
    
    evaluation = Assignment2Evaluation(
        problem=problem, 
        config=config.experiments.algorithm,
        maximize=config.experiments.dataset.problem.maximize
        )

    evaluation.fitness_curve()
    

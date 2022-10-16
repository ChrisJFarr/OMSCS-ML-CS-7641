import six
import sys
sys.modules['sklearn.externals.six'] = six
from src.evaluation import Assignment2Evaluation
from time import perf_counter


def vc(config):
    start_time = perf_counter()
    evaluation = Assignment2Evaluation(config=config)
    evaluation.validation_curve()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))


def fc(config):  # shorthand: fitness-curve
    start_time = perf_counter()
    evaluation = Assignment2Evaluation(config=config)
    evaluation.fitness_curve()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))

def wall(config):
    evaluation = Assignment2Evaluation(config=config)
    evaluation.wall_clock()

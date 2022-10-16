from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')


class Assignment2Evaluation:

    def __init__(self, config):
        # self.problem = problem
        self.config = config
        self.maximize = config.experiments.dataset.problem.maximize
        return
    
    def validation_curve(self):
        # Perform optimization at multiple steps of a parameter value range
        # Loop over hyper-parameters specified in experiments config
        validation_curve_dict = self.config.experiments.evaluation.validation_curve
        for param_name, v in validation_curve_dict.items():
            # Build plot lists
            x_axis = list()

            def my_hydra_utils_call(**kwargs):
                import six
                import sys
                sys.modules['sklearn.externals.six'] = six
                import warnings
                warnings.filterwarnings('ignore')
                return hydra.utils.call(**kwargs)

            data = Parallel(n_jobs=cpu_count())(
                delayed(my_hydra_utils_call)(
                    config=self.config.experiments.algorithm,
                    **{param_name: (param_value, x_axis.append(param_value))[0]}
                    )
                    for param_value in np.arange(**v)
                )
            y_axis = list(map(lambda x: x[1], data))  # Get (_, best_score)[1]

            self._save_validation_curve(
                x_axis=x_axis, 
                y_axis=y_axis, 
                title="Validation Curve", 
                x_label=param_name, 
                save_name="%s_validation_curve_plot.png" % param_name,
                maximize=self.maximize
                )

    def fitness_curve(self):
        # Perform a single fit opteration and plot the fitness curve
        _, best_score, curve = hydra.utils.call(
                config=self.config.experiments.algorithm,
                curve=True
                )
        self._save_fitness_curve(
            x_axis=list(range(len(curve))), 
            y_axis=curve, 
            x_label="Iterations", 
            title="Fitness Curve", 
            save_name="fitness_curve_plot.png"
            )
        return

    def wall_clock(self):
        ITERATIONS = 100
        # Measure clocktime per iteration
        start_time = perf_counter()
        _, _, curve = hydra.utils.call(
                config=self.config.experiments.algorithm,
                curve=True,
                max_iters=ITERATIONS
                )
        end_time = perf_counter()
        print("Clock time per iter: %.6f" % ((end_time - start_time) / len(curve)))

    def _save_fitness_curve(self, x_axis, y_axis, title, x_label, save_name):

        # if (maximize is not None) and (not maximize):
        #     # TODO starts with smallest number is best
        #     # TODO Flip sign, then largest number is best
        #     y_axis = np.array(y_axis) * -1

        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        df = pd.DataFrame(data=y_axis, index=x_axis)
        df.plot(ax=ax, lw=1.25, legend=False)
        # ax.plot(x_axis, df.values, lw=1.25, label=df.columns)
        # plt.ylim((0.5, 1.01))
        plt.xlabel(x_label)
        plt.ylabel("Fitness Score")

        # leg = plt.legend()

        best_score = np.max(y_axis)
        best_i = np.argmax(y_axis)

        plt.axhline(y=best_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=x_axis[best_i], color='black', lw=.5, linestyle="--")

        model_name = self.config.experiments.algorithm._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.problem.fitness_fn._target_.split(".")[-1]

        plt.title(
            "%s(%s, %s)\nfitness: %.3f; maximize: %s; %s=%.3f; " % \
                (title, 
                model_name, 
                dataset_name, 
                best_score * (-1 if not self.maximize else 1), 
                str(self.maximize).lower(),
                x_label,
                x_axis[best_i]),
            fontsize="large"
            )

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))

    def _save_validation_curve(self, x_axis, y_axis, title, x_label, save_name, maximize=None):

        if (maximize is not None) and (not maximize):
            # TODO starts with smallest number is best
            # TODO Flip sign, then largest number is best
            y_axis = np.array(y_axis) * -1

        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        df = pd.DataFrame(data=y_axis, index=x_axis)
        df.plot(ax=ax, lw=1.25, legend=False)
        # ax.plot(x_axis, df.values, lw=1.25, label=df.columns)
        # plt.ylim((0.5, 1.01))
        plt.xlabel(x_label)
        plt.ylabel("Fitness Score")

        # leg = plt.legend()

        best_score = np.max(y_axis)
        best_i = np.argmax(y_axis)
        # print("best_score", best_score)

        plt.axhline(y=best_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=x_axis[best_i], color='black', lw=.5, linestyle="--")

        model_name = self.config.experiments.algorithm._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.problem.fitness_fn._target_.split(".")[-1]

        plt.title(
            "%s(%s, %s)\nfitness: %.3f; maximize: %s; %s=%.3f; " % \
                (title, 
                model_name, 
                dataset_name, 
                best_score * (-1. if not self.maximize else 1.), 
                str(self.maximize).lower(),
                x_label,
                x_axis[best_i]),
            fontsize="large"
            )

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
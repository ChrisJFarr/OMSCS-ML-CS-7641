from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
import os



class Assignment2Evaluation:

    def __init__(self, problem, config, maximize):
        self.problem = problem
        self.config = config
        self.maximize = maximize
        return

    
    def evaluation_curve():
        # Perform optimization at multiple steps of a parameter value range
        raise NotImplementedError

    def fitness_curve(self):
        # Perform a single fit opteration and plot the fitness curve
        best_state, best_score, curve = hydra.utils.call(
                config=self.config,
                problem=self.problem, 
                curve=True
                )
        self._save_line_graph_plot(
            x_axis=list(range(len(curve))), 
            y_axis=curve, 
            title="test title", 
            x_label="Iterations", 
            save_name="test_plot",
            maximize=self.maximize
            )
        return


    def _save_line_graph_plot(self, x_axis, y_axis, title, x_label, save_name, maximize):
        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        # pd.DataFrame(data=y_axis, index=x_axis).sort_index(axis=1).plot(lw=1.25, ax=ax)  # , color="black"
        df = pd.DataFrame(y_axis)
        ax.plot(x_axis, df.values, lw=1.25, label=df.columns)
        # plt.ylim((0.5, 1.01))
        plt.xlabel(x_label)
        plt.ylabel("Fitness")
        # model_name = self.config.experiments.model._target_.split(".")[-1]
        # dataset_name = self.config.experiments.dataset.name.split("_")[1]

        # for line in ax.get_lines():
        #     if "test" in line.get_label():
        #         line.set_linewidth(2.25)
        #         line.set_linestyle("--")

        
        # make a legend for both plots
        leg = plt.legend()
        # set the linewidth of each legend object
        # for legobj in leg.legendHandles:
        #     if "test" in legobj.get_label():
        #         # legobj.set_linewidth(3.0)
        #         legobj.set_linestyle("--")

        # Draw top performance for test
        # This only works with a single metric
        # test_performance = np.array([v for d in y_axis for k, v in d.items() if "test" in k])
        # train_performance = np.array([v for d in y_axis for k, v in d.items() if "train" in k])
        best_score = np.max(y_axis)
        best_i = np.argmax(y_axis)
        # max_test_score = np.max(test_performance)
        # max_test_i = np.argmax(test_performance)
        # train_at_max_test = train_performance[max_test_i]
        # plt.title(
        #     "%s %s; dataset: %s\ntrain AUC: %.3f; test AUC: %.3f; param=%.3f" % \
        #         (model_name, title, dataset_name, train_at_max_test, max_test_score, x_axis[max_test_i]),
        #     fontsize="large"
        #     )
        plt.axhline(y=best_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=x_axis[best_i], color='black', lw=.5, linestyle="--")

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
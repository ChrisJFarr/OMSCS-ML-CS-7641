"""
One evaluation approach for all models/experiments

* Select metrics for classification
* Produce visuals
"""
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import hydra
import os
from hydra.utils import get_original_cwd

from src.models import ModelParent
from src.datamodules import DataParent


# def get_metrics(y_true, y_pred):
#     # Compute sensitivity and specificity
#     score_report = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
#     # precision recall f-measure support
#     sensitivity = score_report[1][1]
#     specificity = score_report[1][0]
#     return {"sensitivity": sensitivity, "specificity": specificity}

def get_metrics(y_true, y_pred):
    # Compute auc score
    roc_auc = roc_auc_score(y_true, y_pred)
    return {"roc_auc": roc_auc}


class Assignment1Evaluation:
    def __init__(self, model: ModelParent, datamodule: DataParent, config):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        return

    def run(self, *args, **kwargs):

        # Produce all visuals needed for model/data combination
        # Evaluate train/test performance, print in summary/log

        # Produce line graphs
        # Across x-axis is dataset size
        # Across y-axis are metric values
        # Include metrics: sensitivity, specificity
        # Compute wall clock time

        pass

    def evaluate(self, train_generator):
        """
        Use model and train-generator to perform train and test evaluation
        """
        # Obtain performance metrics for train and test predictions
        train_performance = list()
        test_performance = list()
        for (x_train, y_train), (x_test, y_test) in train_generator:
            self.model.fit(x_train, y_train)
            train_prediction = self.model.predict_proba(x_train)
            test_prediction = self.model.predict_proba(x_test)
            train_performance.append(get_metrics(y_train, train_prediction))
            test_performance.append(get_metrics(y_test, test_prediction))
        # Compute average performance over splits
        train_avg = pd.DataFrame(train_performance).mean(axis=0).add_suffix('_train').to_dict()
        test_avg = pd.DataFrame(test_performance).mean(axis=0).add_suffix('_test').to_dict()
        return {**train_avg, **test_avg}

    def _save_plot(self, x_axis, y_axis, title, x_label, save_name):
        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        pd.DataFrame(data=y_axis, index=x_axis).sort_index(axis=1).plot(lw=1.0, ax=ax, color="black")
        plt.ylim((0.5, 1.01))
        plt.xlabel(x_label)
        model_name = self.config.experiments.model._target_.split(".")[-1]
        dataset_name = self.config.dataset.name.split("_")[1]
        plt.title("%s %s\n Dataset %s" % (model_name, title, dataset_name))

        for line in ax.get_lines():
            if "test" in line.get_label():
                line.set_linewidth(2.)
                line.set_linestyle("--")
        
        # make a legend for both plots
        leg = plt.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            if "test" in legobj.get_label():
                legobj.set_linewidth(2.0)
                legobj.set_linestyle("--")

        plt.savefig(os.path.join(self.config.run_dir, save_name))
        return

    def generate_learning_curve(self):
        """
        * learning curve plots (some notion of a learning curve)
            * y axis: performance metric
            * x axis: percentage of training data
        """
        x_axis = list()
        y_axis = list()
        # Loop over percentages 10%...100% by 5-10% increments
        for percent in np.arange(.10, 1.01, .10):
            # Track percentages for x-axis
            x_axis.append(percent)
            # Get generator
            train_generator, _ = self.datamodule.make_loader(percent)
            y_axis.append(self.evaluate(train_generator))
        self._save_plot(x_axis, y_axis, title="Learning Curve", x_label="Train Percentage", save_name="learning_curve_plot.png")
        return

    def generate_iterative_plot(self):
        """
        * iterative plot, plot loss curve
            * nn: plot loss for each iteration:
                * y axis loss
                * x axis iteration/epoch
        
        """
        raise NotImplementedError

    def generate_validation_curve(self):
        """
        * validation curve
            * y-axis: performance metric
            * x-axis: varying hyperparameter values
            * plot performance across hyper parameter tuning
        """
        # Loop over hyper-parameters specified in experiments config
        validation_curve_dict = self.config.experiments.evaluation.validation_curve
        for param_name, v in validation_curve_dict.items():
            default_params = {k:v for k, v in self.model.get_params().items()}
            # Build plot lists
            x_axis = list()
            y_axis = list()
            for param_value in np.arange(**v):
                # Set parameter in model
                self.model.set_params(**{param_name: param_value})
                # Track percentages for x-axis
                x_axis.append(param_value)
                # Get generator
                train_generator, _ = self.datamodule.make_loader()
                y_axis.append(self.evaluate(train_generator))
            self._save_plot(
                x_axis, 
                y_axis, 
                title="Validation Curve", 
                x_label=param_name, 
                save_name="%s_validation_curve_plot.png" % param_name
                )
            # Reset model to default parameters
            self.model.set_params(**default_params)

    def evaluate_clock_time(self):
        """
        * wall-clock time
        * fit times during training
        """
        pass

    def evaluate_train_test_performance(self):
        pass

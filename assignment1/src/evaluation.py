"""
One evaluation approach for all models/experiments

* Select metrics for classification
* Produce visuals
"""
from sklearn.metrics import precision_recall_fscore_support

from src.models import ModelParent
from src.datamodules import DataParent


def get_metrics(y_true, y_pred):
    # Compute sensitivity and specificity
    score_report = precision_recall_fscore_support(y_true, y_pred)
    # precision recall f-measure support
    sensitivity = score_report[1][1]
    specificity = score_report[1][0]
    return {"sensitivity": sensitivity, "specificity": specificity}


class Assignment1Evaluation:
    def __init__(self, *args, **kwargs):

        pass

    def run(self, *args, **kwargs):

        # Produce all visuals needed for model/data combination
        # Evaluate train/test performance, print in summary/log

        # Produce line graphs
        # Across x-axis is dataset size
        # Across y-axis are metric values
        # Include metrics: sensitivity, specificity
        # Compute wall clock time

        pass

    def generate_learning_curve(self, model: ModelParent, data: DataParent):
        """
        * learning curve plots (some notion of a learning curve)
            * y axis: performance metric
            * x axis: percentage of training data
        """

        # TODO Start here!

        pass

    def generate_iterative_plot(self):
        """
        * iterative plot, plot loss curve
            * nn: plot loss for each iteration:
                * y axis loss
                * x axis iteration/epoch
        
        """
        pass

    def generate_validation_curve(self):
        """
        * validation curve
            * y-axis: performance metric
            * x-axis: varying hyperparameter values
            * plot performance across hyper parameter tuning
        """
        pass

    def evaluate_clock_time(self):
        """
        * wall-clock time
        * fit times during training
        """
        pass

    def evaluate_train_test_performance(self):
        pass

    def get_metrics(self, y_true, y_pred):
        """
        Produce a dictionary with all tracked metrics for inputs
        """
        metrics = dict()
        return metrics
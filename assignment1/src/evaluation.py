"""
One evaluation approach for all models/experiments

* Select metrics for classification
* Produce visuals
"""
import os
from pprint import pprint
from time import perf_counter

import numpy as np
import pandas as pd
import hydra

from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from joblib import Parallel, delayed
from multiprocessing import cpu_count

from src.models import ModelParent
from src.datamodules import DataParent

import warnings
warnings.filterwarnings('ignore')


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

    def perform_grid_search(self):
        """
        This is only compatible with the Sklearn models
        Print the optimal parameters then return them
        """
        # Get full dataset
        x_train, _, y_train, _ = self.datamodule.get_full_dataset()
        # Create cv
        cv = self.datamodule.get_cv()
        # Build param dict
        params = dict()
        grid_search_params_dict = self.config.experiments.evaluation.grid_search
        grid_search_params_dict = {"model__" + k: v for k, v in grid_search_params_dict.items()}
        for param_name, v in grid_search_params_dict.items():
            params[param_name] = [val for val in np.arange(**v)]
        print("Grid-Search Parameters:")
        pprint(params)
        # Build pipeline
        pipe = Pipeline(
            steps=[
                ("preprocess", self.datamodule),
                ("model", self.model.load().model)
            ]
        )
        # Run grid-search
        grid = GridSearchCV(
            pipe, 
            params,
            scoring=make_scorer(roc_auc_score, needs_proba=True),
            cv=cv,
            n_jobs=cpu_count()-1
            )
        print("Fitting on %s train examples..." % x_train.shape[0])
        grid.fit(x_train, y_train)
        print("Best parameters")
        pprint(grid.best_params_)
        pprint("Best performance: %.3f" % grid.best_score_)
        return grid.best_params_
    
    def cv_score(self):
        # Using the base parameters compute a cross-validated train and test score
        # The test score should be identical to the grid-search test score
        pprint(self.parallel_evaluate(self.datamodule.make_loader()[0]))
        return

    def evaluate(self, train_generator):
        """
        Use model and train-generator to perform train and test evaluation
        """
        # Obtain performance metrics for train and test predictions
        train_performance = list()
        test_performance = list()
        for (x_train, y_train), (x_test, y_test) in train_generator:
            self.model.set_input_size(x_train.shape[1])
            self.model.fit(x_train, y_train)
            train_prediction = self.model.predict_proba(x_train)
            test_prediction = self.model.predict_proba(x_test)
            train_performance.append(get_metrics(y_train, train_prediction))
            test_performance.append(get_metrics(y_test, test_prediction))
        # Compute average performance over splits
        train_avg = pd.DataFrame(train_performance).mean(axis=0).add_suffix('_train').to_dict()
        test_avg = pd.DataFrame(test_performance).mean(axis=0).add_suffix('_test').to_dict()
        return {**train_avg, **test_avg}
    
    def parallel_evaluate(self, train_generator):
        """
        Use model and train-generator to perform train and test evaluation
        """

        def _fit_func(data, model):
            (x_train, y_train), (x_test, y_test) = data
            model.set_input_size(x_train.shape[1])
            model.load().fit(x_train, y_train)
            train_prediction = model.predict_proba(x_train)
            test_prediction = model.predict_proba(x_test)
            train_metric = get_metrics(y_train, train_prediction)
            test_metric = get_metrics(y_test, test_prediction)
            return train_metric, test_metric

        # Obtain performance metrics for train and test predictions
        results = Parallel(n_jobs=cpu_count())(delayed(_fit_func)(data, self.model) for data in train_generator)
        train_performance = list()
        test_performance = list()
        for train_metric, test_metric in results:
            train_performance.append(train_metric)
            test_performance.append(test_metric)

        # Compute average performance over splits
        train_avg = pd.DataFrame(train_performance).mean(axis=0).add_suffix('_train').to_dict()
        test_avg = pd.DataFrame(test_performance).mean(axis=0).add_suffix('_test').to_dict()
        return {**train_avg, **test_avg}

    def _save_plot(self, x_axis: list, y_axis: list, title: str, x_label: str, save_name: str):
        if isinstance(x_axis[0], str):
            return self._save_bar_graph_plot(x_axis, y_axis, title, x_label, save_name)
        elif x_label=="Epoch":
            return self._save_iterative_plot(x_axis, y_axis, title, x_label, save_name)
        else:
            return self._save_line_graph_plot(x_axis, y_axis, title, x_label, save_name)

    def _save_line_graph_plot(self, x_axis, y_axis, title, x_label, save_name):
        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        # pd.DataFrame(data=y_axis, index=x_axis).sort_index(axis=1).plot(lw=1.25, ax=ax)  # , color="black"
        df = pd.DataFrame(y_axis)
        ax.plot(x_axis, df.values, lw=1.25, label=df.columns)
        plt.ylim((0.5, 1.01))
        plt.xlabel(x_label)
        plt.ylabel("AUC")
        model_name = self.config.experiments.model._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.name.split("_")[1]

        for line in ax.get_lines():
            if "test" in line.get_label():
                line.set_linewidth(2.25)
                line.set_linestyle("--")

        
        # make a legend for both plots
        leg = plt.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            if "test" in legobj.get_label():
                # legobj.set_linewidth(3.0)
                legobj.set_linestyle("--")

        # Draw top performance for test
        # This only works with a single metric
        test_performance = np.array([v for d in y_axis for k, v in d.items() if "test" in k])
        train_performance = np.array([v for d in y_axis for k, v in d.items() if "train" in k])
        max_test_score = np.max(test_performance)
        max_test_i = np.argmax(test_performance)
        train_at_max_test = train_performance[max_test_i]
        plt.title(
            "%s %s; dataset: %s\ntrain AUC: %.3f; test AUC: %.3f; param=%.3f" % \
                (model_name, title, dataset_name, train_at_max_test, max_test_score, x_axis[max_test_i]),
            fontsize="large"
            )
        plt.axhline(y=max_test_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=x_axis[max_test_i], color='black', lw=.5, linestyle="--")

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
        return

    def _save_bar_graph_plot(self, x_axis, y_axis, title, x_label, save_name):
        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        sorted_i = np.argsort(x_axis)
        y_axis = list(np.array(y_axis)[sorted_i])
        x_axis = list(np.array(x_axis)[sorted_i])
        pd.DataFrame(data=y_axis, index=x_axis).plot.bar()
        plt.ylim((0.5, 1.01))
        plt.ylabel("AUC")
        plt.xlabel(x_label)
        plt.xticks(rotation=45)
        model_name = self.config.experiments.model._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.name.split("_")[1]
        plt.title("%s %s\n Dataset %s" % (model_name, title, dataset_name))

        test_performance = np.array([v for d in y_axis for k, v in d.items() if "test" in k])
        train_performance = np.array([v for d in y_axis for k, v in d.items() if "train" in k])
        max_test_score = np.max(test_performance)
        max_test_i = np.argmax(test_performance)
        train_at_max_test = train_performance[max_test_i]
        plt.title(
            "%s %s; dataset: %s\ntrain AUC: %.3f; test AUC: %.3f; param=%s" % \
                (model_name, title, dataset_name, train_at_max_test, max_test_score, x_axis[max_test_i]),
            fontsize="large"
            )
        plt.axhline(y=max_test_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=max_test_i, color='black', lw=.5, linestyle="--")

        # make a legend for both plots
        leg = plt.legend()
        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
        return

    def _save_iterative_plot(self, x_axis, y_axis, title, x_label, save_name):
        # Produce a plot with metrics on y-axis
        fig, ax = plt.subplots()
        # pd.DataFrame(data=y_axis, index=x_axis).sort_index(axis=1).plot(lw=1.25, ax=ax)  # , color="black"
        df = pd.DataFrame(y_axis)
        ax.plot(x_axis, df.values, lw=1.25, label=df.columns)
        plt.ylim((0.3, .46))
        plt.xlabel(x_label)
        plt.ylabel("Loss")
        model_name = self.config.experiments.model._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.name.split("_")[1]

        for line in ax.get_lines():
            if "val" in line.get_label():
                line.set_linewidth(2.25)
                line.set_linestyle("--")
                line.set_color('tab:orange')
            else:
                line.set_color('tab:blue')
        
        # make a legend for both plots
        leg = plt.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            if "val" in legobj.get_label():
                # legobj.set_linewidth(3.0)
                legobj.set_linestyle("--")

        # Draw top performance for test
        # This only works with a single metric
        test_performance = np.array([v for d in y_axis for k, v in d.items() if "val" in k])
        train_performance = np.array([v for d in y_axis for k, v in d.items() if "train" in k])
        min_test_score = np.min(test_performance)
        min_test_i = np.argmin(test_performance)
        train_at_min_test = train_performance[min_test_i]
        plt.title(
            "%s %s; dataset: %s\ntrain Loss: %.3f; valid Loss: %.3f; epoch=%.3f" % \
                (model_name, title, dataset_name, train_at_min_test, min_test_score, x_axis[min_test_i]),
            fontsize="large"
            )
        plt.axhline(y=min_test_score, color='black', lw=.5, linestyle="--")
        plt.axvline(x=x_axis[min_test_i], color='black', lw=.5, linestyle="--")

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
        return


    def _save_iterative_plot_2(self, x_axis, y_axis, title, x_label, save_name):

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

        model_name = self.config.experiments.model._target_.split(".")[-1]
        dataset_name = self.config.experiments.dataset.name.split("_")[1]

        plt.title(
            "%s(%s, %s)\nfitness: %.3f; %s=%.3f; " % \
                (title, 
                model_name, 
                dataset_name, 
                best_score, 
                x_label,
                x_axis[best_i]),
            fontsize="large"
            )

        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))

    def generate_learning_curve(self):
        """
        * learning curve plots (some notion of a learning curve)
            * y axis: performance metric
            * x axis: percentage of training data
        """
        x_axis = list()
        y_axis = list()
        # Loop over percentages 10%...100% by 5-10% increments
        for percent in np.arange(**self.config.experiments.evaluation.learning_curve):
            # Track percentages for x-axis
            x_axis.append(percent)
            # Get generator
            train_generator, _ = self.datamodule.make_loader(percent)
            y_axis.append(self.parallel_evaluate(train_generator))
        
        # y_axis = Parallel(n_jobs=cpu_count())(
        #     delayed(self.evaluate)((self.datamodule.make_loader(percent)[0], x_axis.append(percent))[0])
        #     for percent in np.arange(**self.config.experiments.evaluation.learning_curve)
        #     )

        self._save_plot(x_axis, y_axis, title="Learning Curve", x_label="Train Percentage", save_name="learning_curve_plot.png")
        return

    def generate_iterative_plot(self):
        """
        * iterative plot, plot loss curve
            * nn: plot loss for each iteration:
                * y axis loss
                * x axis iteration/epoch

        Use the full train set

        https://pytorch-lightning.readthedocs.io/en/0.9.0/experiment_reporting.html
        
        """
        # x_axis = list()
        # y_axis = list()
        # # Track percentages for x-axis
        # x_axis.append(percent)
        # Get generator
        x_train, _, y_train, _ = self.datamodule.get_full_dataset()
        self.model.set_input_size(x_train.shape[1])
        self.model.load().fit(x_train, y_train)
        # Load loggs
        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        logs_path = os.path.join(override_dirname, "logs", "logs", "metrics.csv")
        logs = pd.read_csv(logs_path)
        log_summary = logs[["val_loss_epoch", "train_loss_epoch", "epoch"]].rename(
            {
                "val_loss_epoch": "val_loss",
                "train_loss_epoch": "train_loss"
                }, axis=1
            ).groupby("epoch").mean()
        x_axis = list(log_summary.index)
        y_axis = log_summary.to_dict(orient="records")
        self._save_plot(x_axis, y_axis, title="Iterative Plot", x_label="Epoch", save_name="iterative_plot.png")

    def generate_iterative_plot2(self):
        """
        * iterative plot, plot loss curve
            * nn: plot loss for each iteration:
                * y axis loss
                * x axis iteration/epoch

        Use the full train set

        https://pytorch-lightning.readthedocs.io/en/0.9.0/experiment_reporting.html
        
        """
        x_train, _, y_train, _ = self.datamodule.get_full_dataset()
        self.model.set_input_size(x_train.shape[1])
        self.model.load()
        nnet = self.model.fit(x_train, y_train)
        y_axis = nnet.fitness_curve
        x_axis = list(range(len(y_axis)))
        self._save_iterative_plot_2(x_axis, y_axis, title="Iterative Plot", x_label="Epoch", save_name="iterative_plot.png")


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
            default_params = {k:v for k, v in self.model.kwargs.items()}
            # Build plot lists
            x_axis = list()
            y_axis = list()
            for param_value in np.arange(**v):
                # Set parameter in model
                self.model.set_params(**{param_name: param_value})
                # Track percentages for x-axis
                external_param_value = self.model.get_param_value(param_name, param_value)
                x_axis.append(external_param_value)
                # Get generator
                train_generator, _ = self.datamodule.make_loader()
                y_axis.append(self.parallel_evaluate(train_generator))
            self._save_plot(
                x_axis, 
                y_axis, 
                title="Validation Curve", 
                x_label=param_name, 
                save_name="%s_validation_curve_plot.png" % param_name
                )
            # Reset model to default parameters
            self.model.set_params(**default_params)  # Only needed if running serial evaluate?

    def evaluate_train_test_performance(self):
        print("Performing full test evaluation...")
        # Get generator
        x_train, x_test, y_train, y_test = self.datamodule.get_full_dataset()
        self.model.set_input_size(x_train.shape[1])
        self.model.load()
        # Start training timer
        start_time = perf_counter()
        # Compute test prediction
        self.model.fit(x_train, y_train)
        # End timer
        end_time = perf_counter()
        print("Training time: %.3f" % (end_time - start_time))
        train_prediction = self.model.predict_proba(x_train)
        # Start inference timer
        start_time = perf_counter()     
        test_prediction = self.model.predict_proba(x_test)
        # End timer
        end_time = perf_counter()
        print("Inference time: %.3f" % (end_time - start_time))
        test_scores = {
            "train_auc": get_metrics(y_train, train_prediction)["roc_auc"],
            "test_auc": get_metrics(y_test, test_prediction)["roc_auc"]
        }
        # Score test prediction
        print("Final Performance Evaluation:")
        pprint(test_scores)

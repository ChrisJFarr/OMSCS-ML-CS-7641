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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

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

class Assignment3Evaluation:
    
    def __init__(self, model: ModelParent, datamodule: DataParent, config):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        return
    
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

    def _save_silhouette_plot(self, x_axis, y_axis_silhouette, y_axis_auc=None, 
        v_line=None, title=None, x_label=None, save_name=None):
        # v_line is selected number of clusters
        # Reference: https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
        dataset_name = self.config.experiments.dataset.name.split("_")[1]
        fig, ax1 = plt.subplots(1,1)
        plt1 = pd.DataFrame(data=y_axis_silhouette, columns=["silhouette"], index=x_axis).plot(ax=ax1, lw=2.0, linestyle="--", color="green")
        
        max_silhouette_i = np.argmax(y_axis_silhouette)
        v_line = x_axis[max_silhouette_i]
        max_silhouette_score = y_axis_silhouette[max_silhouette_i]
        
        if y_axis_auc is not None:
            ax2 = ax1.twinx()
            plt2 = pd.DataFrame(data=y_axis_auc, index=x_axis).plot(ax=ax2)
            # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
            handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.get_legend().remove()
            ax2.get_legend().remove()
            plt.legend(handles, labels)
            plt.ylim((0.5, 1.01))
            test_performance = np.array([v for d in y_axis_auc for k, v in d.items() if "test" in k])
            train_performance = np.array([v for d in y_axis_auc for k, v in d.items() if "train" in k])
            assert v_line in x_axis
            select_i = np.argwhere(np.array(x_axis)==v_line)
            select_test = test_performance[select_i]
            select_train = train_performance[select_i]

            plt.title(
                "%s; dataset: %s\ntrain AUC: %.3f; test AUC: %.3f; n_clusters=%s" % \
                    (title, dataset_name, select_train, select_test, v_line),
                fontsize="large"
                )
            plt.axhline(y=select_test, color='black', lw=.5, linestyle="--")
        else:
            plt.title(
                "%sdataset: %s; n_clusters=%s; silhouette score: %.2f" % \
                    (title, dataset_name, v_line, max_silhouette_score),
                fontsize="large"
                )
        plt.axvline(x=v_line, color='black', lw=.5, linestyle="--")
        plt.xlabel("n-cluster")
        
        if "GaussianMixture" in self.config.experiments.datamodule.cluster[0]._target_:
            plt.ylabel("score")
        else:
            plt.ylabel("silhouette score")
        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, save_name))
        return

    def _save_silhouette_plot_multi(self):
        # TODO Implement
        raise NotImplementedError

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
        x_train, _, y_train, _ = self.datamodule.load().get_full_dataset()
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
        x_train, x_test, y_train, y_test = self.datamodule.load().get_full_dataset()
        x_train = self.datamodule.fit(x_train, y_train).transform(x_train)
        x_test = self.datamodule.transform(x_test)
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
        print("Feature importance:")
        pprint(self.datamodule.get_feature_importance(x_train))

    def random_repeated_cluster_feature_importance(self):
        print("Evaluating feature importance...")
        # Get generator
        train_generator, _ = self.datamodule.make_loader(track_importance=True)
        # Just loop through them real quick
        for _ in train_generator:
            pass
        importance_df = pd.DataFrame(self.datamodule.importance_agg)
        # print(importance_df.sum())
        # print(importance_df.mean())
        # Compute distribution of components selected from random clusters
        importance_df = importance_df.sum() / importance_df.sum().sum()
        # Decide manually if should sort axis or not 
        # importance_df.sort_values(inplace=True, ascending=False)
        importance_df.plot.bar()
        plt.xticks(rotation=65)
        plt.title("Random Cluster Set + Lasso Selection", fontsize="large")
        # TODO Add experiment title
        plt.ylabel("Selection Density")
        plt.tight_layout(pad=2.0)
        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, "feature_importance_bar.png"))

    def silhouette_plot(self):
        self.datamodule.load()
        assert self.datamodule.cluster is not None
        assert len(self.datamodule.cluster) == 1

        # Get clustering params (range of k's)
        silhouette_plot_dict = self.config.experiments.evaluation.silhouette_plot
        for param_name, v in silhouette_plot_dict.items():
            # Build plot lists
            x_axis = list()
            y_axis = list()
            for param_value in np.arange(**v):
                # Set params for the cluster algorithm
                for cluster in self.datamodule.cluster:
                    cluster.set_params(**{param_name: param_value})
                # Get data for scoring
                x_train, _, y_train, _ = self.datamodule.load().get_full_dataset()
                x_train = self.datamodule.fit(x_train, y_train).transform(x_train)
                cluster_labels = x_train.filter(like="cluster_")
                x_train = x_train.drop(cluster_labels.columns, axis=1)
                y_axis.append(silhouette_score(x_train, cluster_labels))
                x_axis.append(param_value)
        plot_name = self.config.experiments.datamodule.plot_name
        self._save_silhouette_plot(
            x_axis, 
            y_axis, 
            y_axis_auc=None,
            title=f"{plot_name} Silhouette Plot\n", 
            x_label=param_name, 
            save_name="silhouette_plot.png"
            )
        
    def silhouette_plot_with_auc(self):
        # TODO Adapt to silhouette
        assert self.datamodule.cluster is not None
        assert len(self.datamodule.cluster) == 1

        # Get clustering params (range of k's)
        silhouette_plot_dict = self.config.experiments.evaluation.silhouette_plot
        for param_name, v in silhouette_plot_dict.items():
            # Build plot lists
            x_axis = list()
            y_axis_silhouette = list()
            y_axis_auc = list()
            for param_value in np.arange(**v):
                # Set params for the cluster algorithm
                for cluster in self.datamodule.cluster:
                    cluster.set_params(**{param_name: param_value})
                # Fit clustering with params
                x_train, _, y_train, _ = self.datamodule.load().get_full_dataset()
                x_train = self.datamodule.fit(x_train, y_train).transform(x_train)
                cluster_labels = x_train.filter(like="cluster_")
                x_train = x_train.drop(cluster_labels.columns, axis=1)
                y_axis_silhouette.append(silhouette_score(x_train, cluster_labels))
                # Get generator
                train_generator, _ = self.datamodule.make_loader()
                # Fit model and get performance too (plot alongside silhouette for experiment)
                y_axis_auc.append(self.parallel_evaluate(train_generator))
                x_axis.append(param_value)
        self._save_silhouette_plot(
            x_axis, 
            y_axis_silhouette, 
            y_axis_auc,
            title="Silhouette Plot", 
            x_label=param_name, 
            save_name="silhouette_auc_plot.png"
            )

    def tsne_plot(self):
        # Sources: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
        # https://scikit-learn.org/stable/modules/manifold.html#t-sne
        x_train, x_test, y_train, y_test = self.datamodule.load().get_full_dataset()
        x_train = self.datamodule.fit(x_train, y_train).transform(x_train)
        if "rotate_projection_180" in self.config.experiments.datamodule:
            rotate_180 = self.config.experiments.datamodule.rotate_projection_180
            if rotate_180:
                x_train *= -1.0
        if "tsne_dot_size" in self.config.experiments.datamodule:
            dot_size = self.config.experiments.datamodule.tsne_dot_size
        else:
            dot_size = 15.0
        tsne_data = TSNE(
                n_components=x_train.shape[1],
                init=x_train.values,
                learning_rate="auto",
                n_iter=500,
                n_iter_without_progress=150,
                n_jobs=2,
                random_state=0,
            ).fit_transform(x_train)
        fig, ax = plt.subplots()
        for l in [0, 1]:
            ax.scatter(
                    x=tsne_data[y_train==l,0], 
                    y=tsne_data[y_train==l,1],
                    s=np.ones_like(y_train[y_train==l]) * dot_size,
                    color=["tab:green", "tab:red"][l],
                    label=["Neg Diabetes", "Pos Diabetes"][l]
                )
        ax.legend()
        ax.grid(True)
        projection_algo = self.config.experiments.datamodule.projection[0]._target_.split(".")[-1]
        plt.title("TSNE + %s" % projection_algo)
        hdr = HydraConfig.get()
        override_dirname= hdr.run.dir
        plt.savefig(os.path.join(override_dirname, "tsne.png"))


    def feature_selection(self):
        # Compute feature selection
        # Print all column names
        # Print selected column names
        # Vary lasso parameter count number of features selected?
        feature_selection_dict = self.config.experiments.evaluation.feature_selection
        for param_name, v in feature_selection_dict.items():
            # Build plot lists
            x_axis = list()
            y_axis = list()
            for param_value in np.arange(**v):
                # Set parameter in model
                self.datamodule.set_params(**{param_name: param_value})
                # Track percentages for x-axis
                x_axis.append(param_value)
                x_train, _, y_train, _ = self.datamodule.load().get_full_dataset()
                x_train = self.datamodule.fit(x_train, y_train).transform(x_train)
                y_axis.append(x_train.shape[1])
            pd.DataFrame(data=y_axis, index=x_axis, columns=["n-features"]).plot(lw=2.0, linestyle="--")
            plt.ylabel("n-features")
            plt.xlabel("Lasso Alpha")
            plt.xticks(rotation=45)
            dataset_name = self.config.experiments.dataset.name.split("_")[1]
            plt.title(
                "Lasso Feature Selection\ndataset: %s" % \
                    (dataset_name, ),
                fontsize="large"
                )
            plt.tight_layout(pad=2.0)
            hdr = HydraConfig.get()
            override_dirname= hdr.run.dir
            plt.savefig(os.path.join(override_dirname, "feature_importance.png"))

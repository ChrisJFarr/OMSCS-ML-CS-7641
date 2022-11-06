from abc import ABC, abstractmethod
from cgi import test
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import (
    MinMaxScaler, Normalizer
    )
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


class DataParent(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.produce_ds2_small = False
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.importance_agg = list()

    @abstractmethod
    def load(self):
        return self

    def get_full_dataset(self):
        # Load csv
        df = self._load_data()
        x_data = df.drop(self.target, axis=1)
        y_data = df[self.target]
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=self.test_size, 
            stratify=y_data, 
            random_state=self.seed
            )
        # Sample to 10% of original, for KNN only
        if self.produce_ds2_small:
            df = pd.concat([x_train, y_train], axis=1)
            df = df.sample(frac=.1, random_state=self.seed)
            x_train = df.drop(self.target, axis=1)
            y_train = df[self.target]
        return x_train, x_test, y_train, y_test

    def make_loader(self, train_percentage=1.0, track_importance=False):
        # Load the train/test dataset
        x_train, x_test, y_train, y_test = self.get_full_dataset()
        # Select percentage of training data
        train = pd.concat([x_train, y_train], axis=1).sample(frac=train_percentage, random_state=self.seed)
        x_train, y_train = train.iloc[:, :x_train.shape[1]], train.iloc[:, -1]
        # Return train generator and test set
        return (self._cv_generator(x_train, y_train, track_importance=track_importance), (x_test, y_test))

    def get_cv(self):
        return RepeatedStratifiedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=self.seed)

    def _cv_generator(self, x_data, y_data, track_importance=False):
        # Stratified k-fold CV
        kfold = self.get_cv()
        if track_importance:
            self.importance_agg = list()
        for train, test in kfold.split(x_data, y_data):
            cpy = deepcopy(self).load()

            x_train_cpy = x_data.iloc[train].copy()
            y_train_cpy = y_data.iloc[train].copy()
            x_test_cpy = x_data.iloc[test].copy()
            y_test_cpy = y_data.iloc[test].copy()
            x_train_cpy = cpy.fit(x_train_cpy, y_train_cpy).transform(x_train_cpy)
            if track_importance:
                self.importance_agg.append(deepcopy(cpy.get_feature_importance(x_train_cpy)))
            x_test_cpy = cpy.transform(x_test_cpy)
            # print(x_train_cpy.head())

            # Yield results
            yield ((x_train_cpy, y_train_cpy), (x_test_cpy, y_test_cpy))
    
    def _load_data(self):
        """
        Handle any data-specific preprocessing here
        """
        # print("dataset", print(self.keys()))
        df = pd.read_csv(self.path)
        # For dataset=_2_, if contains unique values {0,1,2} convert to {0, 1 if 1 or 2}
        if self.name=="_2_":
            df.loc[:, self.target] = df[self.target].apply(lambda x: int(x>0)).values
        return df

    def fit(self, x_train, y_train=None):
        x_train = x_train.copy() 
        x_train = self.scale_or_normalize(x_train=x_train)
        x_train = self.feature_projection(x_train=x_train)
        steps = (self.add_features, self.select_features)
        step_order = [0, 1]
        try:
            if hasattr(self, 'select_first') & self.select_first:
                print("feature selection first...")
                step_order = [1, 0]
        except Exception:
            pass
        for step in step_order:
            x_train = steps[step](x_train=x_train, y_train=y_train)
        # x_train = self.add_features(x_train=x_train, y_train=y_train)
        # x_train = self.select_features(x_train=x_train, y_train=y_train)
        # TODO Print or store nested feature importance
        return self

    def transform(self, x_data):
        x_data = x_data.copy()
        x_data = self.scale_or_normalize(x_test=x_data)
        x_data = self.feature_projection(x_test=x_data)
        steps = (self.add_features, self.select_features)
        step_order = [0, 1]
        try:
            if hasattr(self, 'select_first') & self.select_first:
                step_order = [1, 0]
        except Exception:
            pass
        for step in step_order:
            x_data = steps[step](x_test=x_data)
        # x_data = self.add_features(x_test=x_data)
        # x_data = self.select_features(x_test=x_data)
        return x_data

    @abstractmethod
    def scale_or_normalize(self, x_train, x_test):
        pass

    @abstractmethod
    def feature_projection(self, x_train, x_test):
        pass

    @abstractmethod
    def add_features(self, x_train, y_train, x_test):
        pass

    @abstractmethod
    def select_features(self, x_train, y_train, x_test):
        pass

    def set_params(self, **params):
        for k, v in params.items():
            self.kwargs[k] = v


class NeuralNetworkData(DataParent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = deepcopy(kwargs)
        self.cluster = None
        self.projection = None
        self.random_clusters = 0
        self.normalize_clusters = False
        self.select_from_model = False
        self.lasso_alpha = 0.001
        self.random_columns = []
        self.fitted_clusters = []
        self.scaler = None
        self.is_loaded = False

    def load(self):
        for k, v in self.kwargs.items():
            setattr(self, k, v)
        if "n_components" in self.kwargs:
            self.projection[0].set_params(**{"n_components": self.kwargs.get("n_components")})
        self.scaler = MinMaxScaler()
        if self.select_from_model:
            self.feature_selector = SelectFromModel(
                estimator=Lasso(alpha=self.lasso_alpha)
                )
        self.random_columns = []
        self.fitted_clusters = []
        self.is_loaded = True
        return self
    
    def get_feature_importance(self, x_data):
        cluster_cols = [col for col in x_data.columns if "cluster_" in col]
        if cluster_cols == []:
            # print(x_data.columns)
            print("WARNING: no clusters present in x-data")
            return False
        cluster_indices = [int(col.split("_")[-1]) for col in cluster_cols]
        column_presence = [col for i in cluster_indices for col in self.random_columns[i]]
        return dict(zip(*np.unique(column_presence, return_counts=True)))
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"

        if x_test is not None:
            # TODO If self.cluster is not None, 
            # TODO then self.cluster is a list, so add a "cluster_%s" column for each clustering algorithm
            # TODO run transform
            if self.cluster is not None:
                for i, cluster in enumerate(self.cluster):
                    if self.random_clusters > 0:
                        for c in range(self.random_clusters):
                            assert x_test is not None
                            x_test["cluster_%s" % c] = self.fitted_clusters[c].predict(x_test.loc[:, self.random_columns[c]])
                            if self.normalize_clusters:
                                x_test.loc[:, "cluster_%s" % c] /= x_test.loc[:, "cluster_%s" % c].max()
                        # TODO Testing new approach dropping original components
                        x_test = x_test.filter(like="cluster_", axis=1)
                    else:
                        x_test["cluster_%s" % i] = cluster.predict(x_test)
                        assert x_test["cluster_%s" % i].std() > 0
                        if self.normalize_clusters:
                            x_test.loc[:, "cluster_%s" % i] /= x_test.loc[:, "cluster_%s" % i].max()
            return x_test

        elif x_train is not None:
            # TODO If self.cluster is not None, 
            # TODO then self.cluster is a list, so add a "cluster_%s" column for each clustering algorithm
            # TODO run fit-transform
            if self.cluster is not None:
                for i, cluster in enumerate(self.cluster):
                    if self.random_clusters > 0:
                        self.random_columns = []
                        np.random.seed(self.seed)
                        for c in range(self.random_clusters):
                            # Select random columns
                            available_cols = np.array(x_train.columns) if c == 0 else np.array(x_train.columns)[:-c]
                            select_cols = np.random.choice(
                                available_cols, 
                                size=2, 
                                replace=False,
                                )
                            # fit cluster
                            # cluster = deepcopy(self.cluster[0]).set_params(**{"n_clusters": len(select_cols) // 2}).fit(x_train.loc[:, select_cols])
                            cluster = KMeans(self.cluster[0].get_params()).set_params(**{"n_clusters": 2}).fit(x_train.loc[:, select_cols])
                            # add columns to random_columns
                            self.random_columns.append(select_cols)
                            # add cluster object to fitted_clusters
                            self.fitted_clusters.append(cluster)
                            # Add cluster to dataset
                            new_clusters = self.fitted_clusters[c].predict(x_train.loc[:, self.random_columns[c]])
                            if new_clusters.sum() == 0:
                                continue
                            x_train["cluster_%s" % c] = new_clusters
                            assert not x_train.isna().sum().any(), "1"
                            if self.normalize_clusters:
                                x_train.loc[:, "cluster_%s" % c] /= x_train.loc[:, "cluster_%s" % c].max()
                                assert not x_train.isna().sum().any(), "2"
                        # TODO New idea: keep only the cluster features here
                        x_train = x_train.filter(like="cluster_", axis=1)
                    else:
                        # x_train = x_train.reset_index(drop=True)
                        cluster.fit(x_train)
                        # print(cluster)
                        # print(cluster.predict(x_train))
                        x_train["cluster_%s" % i] = cluster.predict(x_train)
                        # print(x_train.head())
                        assert x_train["cluster_%s" % i].std() > 0, "TODO WHy are clusters set to 1?"
                        if self.normalize_clusters:
                            x_train.loc[:, "cluster_%s" % i] /= x_train.loc[:, "cluster_%s" % i].max()
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            if self.select_from_model:
                og_cols = list(x_test.columns)
                x_test = pd.DataFrame(data=self.feature_selector.transform(x_test))
                x_test.columns = self.feature_selector.get_feature_names_out(og_cols)
                # print("selected columns:", x_test.columns)
            return x_test
        elif x_train is not None:
            if self.select_from_model:
                # print("available columns:", x_train.columns)
                og_cols = list(x_train.columns)
                x_train = pd.DataFrame(data=self.feature_selector.fit_transform(x_train, y_train))
                x_train.columns = self.feature_selector.get_feature_names_out(og_cols)
                # print("selected columns:", x_train.columns)
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"

        if x_test is not None:
            x_test.iloc[:, :] = self.scaler.transform(x_test)
            return x_test

        elif x_train is not None:
            x_train.iloc[:, :] = self.scaler.fit_transform(x_train)
            return x_train

    def feature_projection(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"

        if x_test is not None:
            # TODO If self.reduction is not None, fit/transform all features and return a new df
            if self.projection is not None:
                for projection in self.projection:
                    x_test = pd.DataFrame(projection.transform(x_test))
                    x_test.columns = ["component_%s" % str(c) for c in x_test.columns]
            return x_test

        elif x_train is not None:
            # TODO If self.reduction is not None, fit/transform all features and return a new df
            if self.projection is not None:
                for projection in self.projection:
                    x_train = pd.DataFrame(projection.fit_transform(x_train))
                    x_train.columns = ["component_%s" % str(c) for c in x_train.columns]
            return x_train

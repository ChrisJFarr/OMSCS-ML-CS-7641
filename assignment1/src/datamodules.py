"""
TODO Design data-module interface

* Starting with decision trees, create DecisionTreeData module
* Decide what DataParent should implement
* Decide how to consume experiments yaml files
    * specify data source between 


Design Concepts:
    * general k-fold cross validation loader (generator function)
    * model-specific features, general features
        * add features
        * select features
"""
from abc import ABC, abstractmethod
from cgi import test
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class DataParent(ABC):

    def __init__(self, path: str, target: str, seed: int):
        super().__init__()
        self.path = path
        self.target = target
        self.seed = seed

    def make_loader(self, n_splits=5):
        # Load csv
        df = pd.read_csv(self.path)
        x_data = df.drop(self.target, axis=1)
        y_data = df[self.target]
        # Stratified k-fold CV
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for train, test in kfold.split(x_data, y_data):
            x_train_cpy = x_data.loc[train].copy()
            y_train_cpy = y_data.loc[train].copy()
            x_test_cpy = x_data.loc[test].copy()
            y_test_cpy = y_data.loc[test].copy()
            # add_features
            x_train_cpy, x_test_cpy = self.add_features(x_train_cpy, y_train_cpy, x_test_cpy)
            # select_features
            x_train_cpy, x_test_cpy = self.add_features(x_train_cpy, y_train_cpy, x_test_cpy)
            # scale/normalize
            x_train_cpy, x_test_cpy = self.scale_or_normalize(x_train_cpy, x_test_cpy)
            # Yield results
            yield ((x_train_cpy, y_train_cpy), (x_test_cpy, y_test_cpy))
    
    @abstractmethod
    def add_features(self, x_train, y_train, x_test):
        pass

    @abstractmethod
    def select_features(self, x_train, y_train, x_test):
        pass

    @abstractmethod
    def scale_or_normalize(self, x_train, x_test):
        pass

class DecisionTreeData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        return x_train, x_test

class NeuralNetworkData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        return x_train, x_test


class BoostingData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        return x_train, x_test


class SVMData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        return x_train, x_test


class KNNData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        return x_train, x_test

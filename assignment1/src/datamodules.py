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
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import (
    MinMaxScaler, Normalizer
    )

class DataParent(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.produce_ds2_small = False
        for k, v in kwargs.items():
            setattr(self, k, v)

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
        return x_train, x_test, y_train, y_test

    def make_loader(self, train_percentage=1.0):
        # Load the train/test dataset
        x_train, x_test, y_train, y_test = self.get_full_dataset()
        # Rebalance dataset by downsampling, filter to homogenous sample
        if self.produce_ds2_small:
            df = pd.concat([x_train, y_train], axis=1)
            df = df.loc[df["Sex"]==0, :] # filtering to only female patients
            pos_df = df.loc[df[self.target]==0, :]
            neg_df = df.loc[df[self.target]==1, :]
            # Balance to same class balance and size as ds1
            df = pd.concat(
                    [
                        pos_df.sample(n=268, replace=False, random_state=self.seed), 
                        neg_df.sample(n=500, replace=False, random_state=self.seed)
                    ], axis=0
                ).reset_index(drop=True)
            x_train = df.drop(self.target, axis=1)
            y_train = df[self.target]
        # Select percentage of training data
        train = pd.concat([x_train, y_train], axis=1).sample(frac=train_percentage, random_state=self.seed)
        x_train, y_train = train.iloc[:, :x_train.shape[1]], train.iloc[:, -1]
        # Return train generator and test set
        return (self._cv_generator(x_train, y_train), (x_test, y_test))

    def get_cv(self):
        return RepeatedStratifiedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=self.seed)

    def _cv_generator(self, x_data, y_data):
        # Stratified k-fold CV
        kfold = self.get_cv()
        for train, test in kfold.split(x_data, y_data):
            x_train_cpy = x_data.iloc[train].copy()
            y_train_cpy = y_data.iloc[train].copy()
            x_test_cpy = x_data.iloc[test].copy()
            y_test_cpy = y_data.iloc[test].copy()
            # # fit/transform train
            # x_train_cpy = self.add_features(x_train=x_train_cpy, y_train=y_train_cpy)
            # x_train_cpy = self.select_features(x_train=x_train_cpy, y_train=y_train_cpy)
            # x_train_cpy = self.scale_or_normalize(x_train=x_train_cpy)
            # # transform test
            # x_test_cpy = self.add_features(x_test=x_test_cpy)
            # x_test_cpy = self.select_features(x_test=x_test_cpy)
            # x_test_cpy = self.scale_or_normalize(x_test=x_test_cpy)

            x_train_cpy = self.fit(x_train_cpy, y_train_cpy).transform(x_train_cpy)
            x_test_cpy = self.transform(x_test_cpy)

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
        x_train = self.add_features(x_train=x_train, y_train=y_train)
        x_train = self.select_features(x_train=x_train, y_train=y_train)
        x_train = self.scale_or_normalize(x_train=x_train)
        return self

    def transform(self, x_data):
        x_data = x_data.copy()
        x_data = self.add_features(x_test=x_data)
        x_data = self.select_features(x_test=x_data)
        x_data = self.scale_or_normalize(x_test=x_data)
        return x_data

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
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

class NeuralNetworkData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = MinMaxScaler()
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            x_test.iloc[:, :] = self.scaler.transform(x_test)
            return x_test
        elif x_train is not None:
            x_train.iloc[:, :] = self.scaler.fit_transform(x_train)
            return x_train

class BoostingData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

class SVMData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = Normalizer()
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            x_test.iloc[:, :] = self.scaler.transform(x_test)
            return x_test
        elif x_train is not None:
            x_train.iloc[:, :] = self.scaler.fit_transform(x_train)
            return x_train

class KNNData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = MinMaxScaler()
    
    def add_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def select_features(self, x_train=None, y_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            return x_test
        elif x_train is not None:
            return x_train

    def scale_or_normalize(self, x_train=None, x_test=None):
        assert x_train is not None or x_test is not None, "must pass x_train or x_test"
        if x_test is not None:
            x_test.iloc[:, :] = self.scaler.transform(x_test)
            return x_test
        elif x_train is not None:
            x_train.iloc[:, :] = self.scaler.fit_transform(x_train)
            return x_train

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
from sklearn.preprocessing import MinMaxScaler


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
            # add_features
            x_train_cpy, x_test_cpy = self.add_features(x_train_cpy, y_train_cpy, x_test_cpy)
            # select_features
            x_train_cpy, x_test_cpy = self.select_features(x_train_cpy, y_train_cpy, x_test_cpy)
            # scale/normalize
            x_train_cpy, x_test_cpy = self.scale_or_normalize(x_train_cpy, x_test_cpy)
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
        scaler = MinMaxScaler()
        x_train.iloc[:, :] = scaler.fit_transform(x_train)
        x_test.iloc[:, :] = scaler.transform(x_test)
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
        scaler = MinMaxScaler()
        x_train.iloc[:, :] = scaler.fit_transform(x_train)
        x_test.iloc[:, :] = scaler.transform(x_test)
        return x_train, x_test

class KNNData(DataParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def select_features(self, x_train, y_train, x_test):
        return x_train, x_test

    def scale_or_normalize(self, x_train, x_test):
        scaler = MinMaxScaler()
        x_train.iloc[:, :] = scaler.fit_transform(x_train)
        x_test.iloc[:, :] = scaler.transform(x_test)
        return x_train, x_test

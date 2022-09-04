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


DATA_PATHS = {
    "DHI": "data/diabetes_012_health_indicators_BRFSS2015.csv",
    "D": "data/diabetes.csv"
}


class DataParent:

    def __init__(self):
        pass

    def make_loader(self, n_folds=5):

        # Get data path
        # Load csv
        # Stratified k-fold CV
        
        for _ in range(n_folds):
            yield (([], []), ([], []))


class DecisionTreeData(DataParent):
    def __init__(self):
        super().__init__()
    
    def add_features(self, x_train, y_train, x_test):
        pass

    def select_features(self, x_train, y_train, x_test):
        pass


class NeuralNetworkData(DataParent):
    def __init__(self):
        super().__init__()

    def add_features(self, x_train, y_train, x_test):
        pass

    def select_features(self, x_train, y_train, x_test):
        pass

class BoostingData(DataParent):
    def __init__(self):
        super().__init__()


class SVMData(DataParent):
    def __init__(self):
        super().__init__()


class KNNData(DataParent):
    def __init__(self):
        super().__init__()

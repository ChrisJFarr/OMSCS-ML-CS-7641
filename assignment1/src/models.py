from sklearn.tree import DecisionTreeClassifier
from abc import ABC, abstractmethod

# TODO Is this needed? I could just use the sklearn interface if that would suffice for all...?

class ModelParent(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class DecisionTreeModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

class NeuralNetworkModel(ModelParent):
    def __init__(self):
        super().__init__()


# Recommended to use AdaBoost
class BoostingModel(ModelParent):
    def __init__(self):
        super().__init__()


class SVMModel(ModelParent):
    def __init__(self):
        super().__init__()


class KNNModel(ModelParent):
    def __init__(self):
        super().__init__()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def get_params(self):
        pass


class DecisionTreeModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        return self.model.set_params(**params)

    def get_params(self):
        return self.model.get_params()

class NeuralNetworkModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()


# Recommended to use AdaBoost
class BoostingModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        base_estimator_params = kwargs.pop("base_estimator_params")
        base_estimator = DecisionTreeClassifier(**base_estimator_params)
        self.model = AdaBoostClassifier(base_estimator=base_estimator, *args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        return self.model.set_params(**params)

    def get_params(self):
        return self.model.get_params()


class SVMModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()


class KNNModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()

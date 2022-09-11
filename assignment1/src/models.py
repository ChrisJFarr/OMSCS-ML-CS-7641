from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

# TODO Is this needed? I could just use the sklearn interface if that would suffice for all...?

class ModelParent(ABC):

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def load(self):
        return self
    
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

    @abstractmethod
    def get_param_value(self, param_name, param_value):
        pass

class DecisionTreeModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        
    def load(self):
        args = self.args
        kwargs = self.kwargs
        self.model = DecisionTreeClassifier(*args, **kwargs)
        self.is_loaded = True
        return self

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        if self.is_loaded:
            self.model.set_params(**params)
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return 

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        return param_value


class MyNeuralNetwork(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        self.model = None
        self.build_model(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        layers = kwargs.get("layers")
        nodes = kwargs.get("nodes")
        input_size = kwargs.get("input_size")
        dropout_rate = kwargs.get("dropout_rate")
        assert layers is not None
        assert nodes is not None
        assert input_size is not None
        assert dropout_rate is not None
        # TODO Build special input layer and output layer
        # For layers > 1, add intermediate layers in a loop
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

        model_list = []

        # Add input layer
        model_list.append(("linear_%s" % 0, torch.nn.Linear(input_size, nodes)))
        model_list.append(("batch_norm_%s" % 0, torch.nn.BatchNorm1d(nodes)))
        model_list.append(("relu_%s" % 0, torch.nn.ReLU()))
        model_list.append(("dropout_%s" % 0, torch.nn.Dropout(p=dropout_rate)))

        # Add intermediate layers
        for layer in range(1, layers):
            model_list.append(("linear_%s" % layer, torch.nn.Linear(nodes, nodes)))
            model_list.append(("batch_norm_%s" % layer, torch.nn.BatchNorm1d(nodes)))
            model_list.append(("relu_%s" % layer, torch.nn.ReLU()))
            model_list.append(("dropout_%s" % layer, torch.nn.Dropout(p=dropout_rate)))

        # Add output layer
        model_list.append(("output", torch.nn.Linear(nodes, 1)))

        self.model = torch.nn.Sequential(OrderedDict(model_list))

    def forward(self, x):
        return self.model(x)

class NeuralNetworkModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        
    def load(self):
        args = self.args
        kwargs = self.kwargs
        self.model = MyNeuralNetwork(*args, **kwargs)
        self.is_loaded = True
        return self

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        # Build training loop

        # Split train into train/validation
        # Fit model and minimize loss (use args for patience)
        # Loop a maxiter times or until loss doesn't improve for patience count
        # For each epoch
        # Loop by batch size
        # optimizer.zero_grad()
        # perform forward pass model()
        # compute loss
        # perform back propagation
        # take an optimizer step with optimizer.step()


        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        if self.is_loaded:
            self.model.set_params(**params)
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return 

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        return param_value



class BoostingModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        
    def load(self):
        args = self.args
        kwargs = self.kwargs
        base_estimator_params = kwargs.pop("base_estimator_params")
        base_estimator = DecisionTreeClassifier(**base_estimator_params)
        self.model = AdaBoostClassifier(base_estimator=base_estimator, *args, **kwargs)
        self.is_loaded = True
        return self

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        if self.is_loaded:
            self.model.set_params(**params)
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return 

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        return param_value

class SVMModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kernels = ("linear", "poly", "rbf", "sigmoid")
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        
    def load(self):
        args = self.args
        kwargs = self.kwargs
        self.model = SVC(*args, **kwargs)
        self.is_loaded = True
        return self

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        # Convert integer kernel-type to str
        if "kernel" in params and str(params["kernel"]).isnumeric():
            params["kernel"] = self.kernels[params.pop("kernel")]
        if self.is_loaded:
            self.model.set_params(**params)
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        if param_name=="kernel":
            param_value = self.kernels[param_value]
        return param_value

class KNNModel(ModelParent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.algorithms = ("ball_tree", "kd_tree", "brute")
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        
    def load(self):
        args = self.args
        kwargs = self.kwargs
        self.model = KNeighborsClassifier(*args, **kwargs)
        self.is_loaded = True
        return self

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        return self.model.predict_proba(*args, **kwargs)[:, 1]

    def set_params(self, **params):
        # Convert integer algorithm-type to str
        if "algorithm" in params and str(params["algorithm"]).isnumeric():
            params["algorithm"] = self.algorithms[params.pop("algorithm")]
        if self.is_loaded:
            self.model.set_params(**params)
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        if param_name=="algorithm":
            param_value = self.algorithms[param_value]
        return param_value
from ensurepip import bootstrap
from os import cpu_count
from src.nn_support import MyNeuralNetwork, LitNeuralNetwork, MyDataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from hydra.core.hydra_config import HydraConfig



class ModelParent(ABC):

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_size = None

    def set_input_size(self, input_size):
        self.input_size=input_size

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


class NeuralNetworkModel(ModelParent):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.is_loaded = False
        self.val = self.kwargs.pop("validation_size", 0.2)
        self.seed = self.kwargs.pop("seed", 42)
        self.lr = self.kwargs.pop("lr", 0.001)
        self.patience = self.kwargs.pop("patience", 3)
        self.batch_size = self.kwargs.pop("batch_size", 32)
        self.save_name = self.kwargs.pop("save_name", "need_save_name")

    def load(self):
        pl.seed_everything(self.seed)
        args = self.args
        kwargs = self.kwargs
        assert self.input_size is not None
        self.model = MyNeuralNetwork(*args, **kwargs, input_size=self.input_size)
        self.early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            mode="min", 
            min_delta=0.0, 
            patience=self.patience)
        self.is_loaded = True
        return self

    def get_iterative_history(self):
        # do something with trainer callback
        pass

    def fit(self, *args, **kwargs):
        assert self.is_loaded
        # Set seed
        pl.seed_everything(self.seed)
        # Split train/validation
        x_data, y_data = args
        x_train, x_val, y_train, y_val = train_test_split(
            x_data,
            y_data,
            test_size=self.val, 
            stratify=y_data, 
            random_state=self.seed
            )
        # Create data loaders
        train_loader = DataLoader(
            MyDataset(x_train, y_train), 
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=5,
            multiprocessing_context='fork'
            )
        valid_loader = DataLoader(
            MyDataset(x_val, y_val), 
            batch_size=self.batch_size, 
            drop_last=True, 
            num_workers=5,
            multiprocessing_context='fork')
        # logging
        try:
            hdr = HydraConfig.get()
            override_dirname= hdr.run.dir
            logger = CSVLogger(override_dirname, name="logs", version="logs")
        except ValueError:
            logger = TensorBoardLogger("lightning_logs")
        # model
        model = LitNeuralNetwork(self.model, self.lr)
        # train model
        trainer = pl.Trainer(
            enable_checkpointing=False,
            log_every_n_steps=5,
            deterministic=True, 
            callbacks=[self.early_stop_callback], 
            enable_progress_bar=False,
            logger=logger,
            )
        # import logging
        # logging.getLogger("lightning").setLevel(logging.ERROR)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        # Get logger info
        return model.logger

    def predict(self, *args, **kwargs):
        return self.predict_proba(self, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        assert self.is_loaded
        x_data = torch.tensor(args[0].values, dtype=torch.float32)
        return self.model(x_data).detach().numpy()

    def set_params(self, **params):
        if self.is_loaded:
            raise NotImplementedError
        else:
            for k, v in params.items():
                self.kwargs[k] = v
        return 

    def get_params(self):
        assert self.is_loaded
        return self.model.get_params()

    def get_param_value(self, param_name, param_value):
        return param_value


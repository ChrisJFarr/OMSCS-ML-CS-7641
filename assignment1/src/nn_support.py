import torch
from collections import OrderedDict
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback



class MyNeuralNetwork(torch.nn.Module):
    # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    def __init__(self, *args, **kwargs):
        super(MyNeuralNetwork, self).__init__()
        self.model = None
        self.build_model(*args, **kwargs)

    def build_model(self, *args, layers, nodes, input_size, dropout_rate):

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
        model_list.append(("logits", torch.nn.Linear(nodes, 1)))
        model_list.append(("output", torch.nn.Sigmoid()))

        self.model = torch.nn.Sequential(OrderedDict(model_list))

    def forward(self, x):
        return self.model(x)

class LitNeuralNetwork(pl.LightningModule):
    # https://pytorch-lightning.readthedocs.io/en/stable/model/train_model_basic.html

    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.loss = torch.nn.BCELoss()
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = batch
        x = x.view(x.size(0), -1)  # Needed?
        loss = self.loss(self.model(x), y)
        self.log("train_loss", loss)
        self.logger.experiment.add_scalars('Iterative Loss', 
                                                {'train': loss}, 
                                                global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # Needed?
        loss = self.loss(self.model(x), y)
        self.log("val_loss", loss)
        self.logger.experiment.add_scalars('Iterative Loss', 
                                                {'valid': loss}, 
                                                global_step=self.global_step)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.learning_rate)
        return optimizer

class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data.values
        self.y_data = y_data.values
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x_data[idx, :], dtype=torch.float32), \
            torch.tensor(self.y_data[idx], dtype=torch.float32).unsqueeze(0)


# class MyCallBack(Callback):


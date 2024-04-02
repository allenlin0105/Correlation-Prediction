import math

import torch
from torch import nn, optim
import lightning as L


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, predict, label):
        loss = torch.sqrt(self.mse(predict, label) + self.eps)
        return loss


class CNN(L.LightningModule):
    def __init__(self, image_size: int):
        super().__init__()
        self.save_hyperparameters()

        activation_func = nn.Tanh()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            activation_func,

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2),
            activation_func,
        )

        for _ in range(2):
            image_size -= 3
            image_size = math.floor(image_size / 2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * image_size * image_size, 1)
        self.act = activation_func

        self.loss_func = RMSELoss()
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.act(self.fc(x))
        return x

    def common_step(self, batch):
        images, labels = batch
        predict = self(images)
        loss = self.loss_func(predict, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
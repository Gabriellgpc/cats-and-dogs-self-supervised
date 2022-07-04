from dataset.cats_and_dogs import CatsAndDogsDataset

import pytorch_lightning as pl

import torch
import torchvision
from torch import nn

import lightly.loss as loss
import lightly.data as data
import lightly.models as models
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamProjectionHead
from lightly.models.modules import SimSiamPredictionHead

import os

import click

class SimSiam(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

@click.command()
@click.option('--input_dir', type=str, default='/datasets/dogs-vs-cats/train')
@click.option('--batch-size', type=int, default=128)
@click.option('--epochs', type=int, default=100)
@click.option('--lr', type=float, default=0.01)
@click.option('--momentum', type=float, default=0.9)
@click.option('--input_size', type=int, default=32)
@click.option('--gpus', type=int, default=1)
def main(input_dir, batch_size, epochs, lr, momentum, input_size, gpus):
    # create a dataset from your image folder
    dataset = data.LightlyDataset(input_dir)

    # the collate function applies random transforms to the input images
    collate_fn = data.ImageCollateFunction(input_size=input_size, cj_prob=0.5)


    # build a PyTorch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,                # pass the dataset to the dataloader
        batch_size=batch_size,         # a large batch size helps with the learning
        shuffle=True,           # shuffling is important!
        collate_fn=collate_fn,   # apply transformations to the input images
        num_workers=os.cpu_count(),
        )

    # use a resnet backbone
    resnet = torchvision.models.resnet.resnet18()
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    # build the simsiam model
    model = SimSiam()

    # use the SimSiam loss function
    criterion = loss.SymNegCosineSimilarityLoss()

    trainer = pl.Trainer(max_epochs=epochs, gpus=1)
    trainer.fit(
        model,
        dataloader
    )
if __name__ == '__main__':
    main()
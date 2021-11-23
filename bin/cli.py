from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.train import train as t


import json

import typer

main = typer.Typer()

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", "--learning_rate", help="Learning rate"
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs"
    ),
    batch_size: int = typer.Option(
        ..., "-bs", "--batch_size", help="Batch size "
    ),
):
    t(name, learning_rate, epochs, batch_size)


@main.command()
def infer():
    print("This is where the inference code will go")

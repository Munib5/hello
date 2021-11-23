from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from ser.train import train as t

import json

import git

import typer

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

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
    model: str = typer.Option(
        ..., "-m", "--model", help="Choice of model"
    ),
):
    t(name, learning_rate, epochs, batch_size, model, sha)


@main.command()
def infer():
    print("This is where the inference code will go")

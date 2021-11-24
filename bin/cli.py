from datetime import datetime
from pathlib import Path

import typer
import torch
import git
import json

from ser.train import train as run_train
from ser.infer import infer as run_infer
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize
from ser.visualise import vis, table_runs

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to load."
    ),
    run: str = typer.Option(
        ..., "-r", "--run", help="Timestamp of interest."
    ),
    label: int = typer.Option(
        ..., "-l", "--label", help="Label you want to infer on."
    ),
):
    run_path = Path(RESULTS_DIR/f"{name}/{run}/")

    # TODO load the parameters from the run_path so we can print them out!
    load_params(run_path)

    # Infer!
    pred, certainty, pixels = run_infer(
        run_path,
        label,
        test_dataloader(1, transforms(normalize))
    )

    vis(pred, certainty, pixels)

    table_runs(run_path)
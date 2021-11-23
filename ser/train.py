from pathlib import Path
import torch
from torch import optim
from .model import Net
import torch.nn.functional as F

from .data import train_data, val_data

import typer

import json

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

print(MODEL_DIR)

main = typer.Typer()

def train(name, learning_rate, epochs, batch_size):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs
    batch_size = batch_size
    learning_rate = learning_rate

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # dataloaders
    training_dataloader = train_data(batch_size)

    validation_dataloader = val_data(batch_size)

    print("Data loaded properly.")

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
        # validate
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                model.eval()
                output = model(images)
                val_loss += F.nll_loss(output, labels, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )
    # save the parameters!

    params = {'name':name, 'learning_rate':learning_rate, 'epochs':epochs, 'batch_size':batch_size, 'val_acc': val_acc}

    with open(f'{name}.json','w') as d:
        json.dump(params, d)

    torch.save(model.state_dict(), MODEL_DIR/f"model_{name}")
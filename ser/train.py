from pathlib import Path
import torch
from torch import optim
from .model import Net
from .model import MnistResNet
import torch.nn.functional as F
import torch.nn as nn

from .data import train_data, val_data

import typer

import json

from datetime import datetime

MODELS = {
    "basic": Net,
    "resnet": MnistResNet,
}

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

main = typer.Typer()

def train(name, learning_rate, epochs, batch_size, model, sha):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs
    batch_size = batch_size
    learning_rate = learning_rate

    # load model
    klass = MODELS[model]
    model = klass().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # dataloaders
    training_dataloader = train_data(batch_size)

    validation_dataloader = val_data(batch_size)

    print("Data loaded properly.")

    correct_temp = 0
    epoch_temp = 0
    loss_function = nn.CrossEntropyLoss()

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            #loss = F.nll_loss(output, labels)
            loss = loss_function(output, labels)
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
                #val_loss += F.nll_loss(output, labels, reduction="sum").item()
                val_loss = loss_function(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            if val_acc > correct_temp:
                correct_temp = val_acc
                epoch_temp = i
                print("Validation accuracy improved")
                # save the parameters!
                params = {'hash':sha, 'name':name, 'learning_rate':learning_rate, 'epochs':epochs, 'batch_size':batch_size, 'val_acc': val_acc, 'epoch': epoch}

                dateTimeObj = datetime.now()

                date = dateTimeObj.strftime("%d-%m-%H-%M")

                Path(PROJECT_ROOT/"results"/f"{date}").mkdir(parents=True, exist_ok=True)

                with open(PROJECT_ROOT/"results"/f"{date}"/f'{name}.json','w') as d:
                    json.dump(params, d)

                torch.save(model.state_dict(), PROJECT_ROOT/"results"/f"{date}"/f"model_{name}.pt")

            print(
                f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )

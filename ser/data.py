from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from .transforms import ts

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
    
# dataloaders
def train_data(bs):
    training_dataloader = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts()),
            batch_size=bs,
            shuffle=True,
            num_workers=1,
        )
    return training_dataloader
def val_data(bs):
    validation_dataloader = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts()),
            batch_size=bs,
            shuffle=False,
            num_workers=1,
        )
    return validation_dataloader

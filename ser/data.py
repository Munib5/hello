from torch.utils.data import DataLoader
from torchvision import datasets
from transforms import ts
    
# dataloaders
def train_data(bs):
    training_dataloader = DataLoader(
            datasets.MNIST(root="../data", download=True, train=True, transform=ts),
            batch_size=bs,
            shuffle=True,
            num_workers=1,
        )
    return training_dataloader
def val_data(bs):
    validation_dataloader = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
            batch_size=bs,
            shuffle=False,
            num_workers=1,
        )
    return validation_dataloader

from torchvision import transforms

# torch transforms
def ts():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
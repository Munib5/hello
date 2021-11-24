import torch
import torch.nn.functional as F

from ser.model import Net


def infer(run_path, label, dataloader):
    # select image to run inference for
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    return pred, certainty, pixels

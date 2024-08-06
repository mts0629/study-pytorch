"""Save and load a model with TorchScript."""

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2  # Use transforms.v2 API


class SimpleNN(nn.Module):
    """Simple neural network."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(in_features=28*28, out_features=512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    test_data = datasets.MNIST(
        root="work/data",
        train=False,
        download=True,
        transform=transform
    )

    model = SimpleNN()
    print(model)

    # Get a TorchScript IR
    x = test_data[0][0]  # Input shape is fixed
    traced_model = torch.jit.trace(model, x)

    print(type(traced_model))
    print(traced_model)

    # The same result with nn.Module can be get
    assert torch.equal(model(x), traced_model(x))

    # Graph representation
    print(traced_model.graph)
    # Python-syntax interpretation
    print(traced_model.code)

    # Save the TorchScript module
    # includes code, parameters, attributes and debug information
    SCRIPT_PATH = "work/simplenn.pt"
    traced_model.save(SCRIPT_PATH)

    # Load the module (to CPU)
    pt_model = torch.jit.load(SCRIPT_PATH).to("cpu").eval()
    assert torch.equal(model(x), pt_model(x))

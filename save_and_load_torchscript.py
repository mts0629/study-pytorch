"""Save and load a model with TorchScript."""

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2  # Use transforms.v2 API


def get_tracing():
    """Get a TorchScript tracing."""
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
    TRACE_PATH = "work/simplenn.pt"
    traced_model.save(TRACE_PATH)

    # Load the module (to CPU)
    pt_model = torch.jit.load(TRACE_PATH).to("cpu").eval()
    assert torch.equal(model(x), pt_model(x))


def get_script():
    """Get a TorchScript."""
    class Gate(nn.Module):
        """Gate module, fake RNN."""

        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    class Cell(nn.Module):
        """Cell module, fake RNN."""

        def __init__(self, gate):
            super().__init__()
            self.gate = gate
            self.linear = nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.gate(self.linear(x)) + h)
            return new_h, new_h


    x = torch.rand(3, 4)
    h = torch.rand(3, 4)

    cell = Cell(Gate())
    print(cell)

    traced_cell = torch.jit.trace(cell)
    print(traced_cell.code)


    scripted_gate = torch.jit.script(Gate())
    scripted_cell = torch.jit.script(Cell(scripted_gate))

    print(scripted_gate.code)
    print(scripted_cell.code)



if __name__ == "__main__":
    # get_tracing()

    get_script()

"""Export a model to ONNX format."""

import torch
import torch.nn.functional as F

from torch import nn


class SimpleCNN(nn.Module):
    """ Simple CNN.
    (input shape is thought as (C,H,W)=(3,32,32))
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN()

    # Dummy input
    x = torch.randn(1, 3, 32, 32)

    # TorchDynamo-based ONNX exporter
    onnx_program = torch.onnx.dynamo_export(model, x)
    onnx_program.save("./work/simplecnn_dynamo.onnx")

    # TorchScript-based ONNX exporter
    torch.onnx.export(
        model,
        x,
        "./work/simplecnn_ts.onnx",
        verbose=True,  # Show human-readable representation
        input_names=["input"],
        output_names=["logit"]
    )

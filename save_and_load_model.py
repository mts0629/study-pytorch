"""Train a simple NN with Fashion-MNIST dataset."""

import torch

import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, RandomSampler
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


def train(dataloader, device, model, loss_fn, optimizer):
    """Train the network."""
    data_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    model.train()

    total_loss, corrects = 0, 0
    for x, t in dataloader:
        x, t = x.to(device), t.to(device)

        optimizer.zero_grad()

        y = model(X)
        loss = loss_fn(y, t)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        corrects += (y.argmax(1) == t).type(torch.float).sum().item()

    total_loss /= data_size
    acc = corrects / data_size
    print(
        f"Train accuracy: {(100 * acc):.2f}[%], train loss: {total_loss:.7f}"
    )


def test(dataloader, device, model, loss_fn):
    """Test the network."""
    data_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)

    model.eval()

    loss, corrects = 0, 0
    with torch.no_grad():
        for x, t in dataloader:
            x, t = x.to(device), t.to(device)
            y = model(x)

            loss += loss_fn(y, t).item() * batch_size
            corrects += (y.argmax(1) == t).type(torch.float).sum().item()

    loss /= data_size
    acc = corrects / data_size
    print(f"Test accuracy: {(100 * acc):.2f}[%], test loss: {loss:.7f}")


def evaluate(model, dataset, device, num_samples, class_labels):
    """Evaluate a model with random-sampled data."""
    # Random sampling from a dataset
    sampler = RandomSampler(dataset, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    model.eval()

    corrects = 0
    with torch.no_grad():
        for x, t in dataloader:
            x = x.to(device)
            y = model(x)
            p = y[0].argmax(0).to("cpu")
            predicted, actual = class_labels[p], class_labels[t]
            print(f'Predicted: "{predicted}", actual: "{actual}"')
            if p == t:
                corrects += 1
    print(
        f"Correct: {corrects}/{num_samples} "
        f"({corrects / num_samples * 100:.1f}[%])"
    )


if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    training_data = datasets.FashionMNIST(
        root="work/data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="work/data",
        train=False,
        download=True,
        transform=transform
    )

    BATCH_SIZE = 64
    train_dataloader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training and evaluation
    EPOCHS = 10
    for epoch in range(EPOSHS):
        print(f"--------------------")
        print(f"Epoch {epoch + 1}")
        print(f"--------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    WEIGHT_PATH = "work/weight.pth"
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f'Save weights to "{WEIGHT_PATH}"')

    PKL_PATH = "work/simplenn.pt"
    torch.save(model, PKL_PATH)
    print(f'Save a model to "{PKL_PATH}"')

    print(f"--------------------")
    print(f"Load weights")
    print(f"--------------------")
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    evaluate(
        model,
        test_data,
        device,
        num_samples=10,
        class_labels=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
    )

    print(f"--------------------")
    print(f"Load a model")
    print(f"--------------------")
    model = torch.load(PKL_PATH)

    evaluate(
        model,
        test_data,
        device,
        num_samples=10,
        class_labels=[
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]
    )

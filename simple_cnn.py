"""Train a simple CNN with CIFAR-10 dataset."""

import torch

import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import v2  # Use transforms.v2 API


class SimpleCNN(nn.Module):
    """Simple CNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten(start_dim=1)  # C dim for NCHW
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(dataloader, device, model, loss_fn, optimizer):
    """Train the network."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    print_loss_iter = num_batches // 10

    model.train()

    total_loss, correct = 0, 0
    for i, (x, t) in enumerate(dataloader):
        x, t = x.to(device), t.to(device)

        optimizer.zero_grad()

        y = model(x)
        loss = loss_fn(y, t)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (y.argmax(1) == t).type(torch.float).sum().item()

        if i % print_loss_iter == 0:
            print(
                f"Train loss: {total_loss/(i + 1):.7f} "
                f"({(i + 1)*len(x)}/{size})"
            )

    total_loss /= size
    correct /= size

    return total_loss, correct


def test(dataloader, device, model, loss_fn):
    """Test the network."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    loss, correct = 0, 0
    with torch.no_grad():
        for x, t in test_dataloader:
            x, t = x.to(device), t.to(device)
            y = model(x)

            loss += loss_fn(y, t).item()
            correct += (y.argmax(1) == t).type(torch.float).sum().item()

    loss /= size
    correct /= size
    print(f"Test accuracy: {(100*correct):.2f}[%], test loss: {loss:.7f}")

    return loss, correct


def plot_histories(train_history, test_history, ylabel):
    """Plot train/test histories."""
    assert len(train_history) == len(test_history)

    fig, ax = plt.subplots()

    t = np.linspace(1, len(train_history), len(train_history))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.plot(t, train_history, label="Train")
    ax.plot(t, test_history, label="Test")
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Data transformations
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # Normalize to [-1, 1]
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR10(
        root="work/data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.CIFAR10(
        root="work/data",
        train=False,
        download=True,
        transform=transform
    )

    BATCH_SIZE = 32
    train_dataloader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    # Momentum SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Train/test history
    history = {
        "train": {"losses": [], "accs": []},
        "test": {"losses": [], "accs": []},
    }

    EPOCHS = 20
    for epoch in range(EPOCHS):
        print(f"--------------------")
        print(f"Epoch {epoch + 1}")
        print(f"--------------------")

        train_loss, train_acc = train(
            train_dataloader, device, model, loss_fn, optimizer
        )
        test_loss, test_acc = test(test_dataloader, device, model, loss_fn)

        history["train"]["losses"].append(train_loss)
        history["test"]["losses"].append(test_loss)

        history["train"]["accs"].append(train_acc)
        history["test"]["accs"].append(test_acc)

    # Plot histories
    plot_histories(
        history["train"]["losses"], history["test"]["losses"], "Loss"
    )
    plot_histories(
        history["train"]["accs"], history["test"]["accs"], "Accuracy"
    )

    MODEL_PATH = "work/simplecnn.pth"
    torch.save(model.state_dict(), MODEL_PATH)

    NUM_TEST = 10
    print("------------------------------")
    print(f"Test with random {NUM_TEST} data")
    print("------------------------------")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    # CIFAR-10 classes
    classes = [
        "Plane", "Car", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck",
    ]

    # Random sampling from test dataset
    sampler = RandomSampler(test_data, num_samples=NUM_TEST)
    eval_loader = DataLoader(test_data, batch_size=1, sampler=sampler)

    correct = 0
    with torch.no_grad():
        for x, t in eval_loader:
            x = x.to(device)
            y = model(x)
            p = y[0].argmax(0).to("cpu")
            predicted, actual = classes[p], classes[t]
            print(f'Predicted: "{predicted}", actual: "{actual}"')
            if p == t:
                correct += 1
    print(f"Correct: {correct}/{NUM_TEST} ({correct / NUM_TEST * 100:.1f}[%])")

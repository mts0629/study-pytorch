"""Save the current state of training as a checkpoint."""

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


def train(dataloader, device, model, loss_fn, optimizer):
    """Train the network."""
    data_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    model.train()

    total_loss, corrects = 0, 0
    for x, t in dataloader:
        x, t = x.to(device), t.to(device)

        optimizer.zero_grad()

        y = model(x)
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

    return total_loss, acc


def test(dataloader, device, model, loss_fn):
    """Test the network."""
    data_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

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


if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    training_data = datasets.MNIST(
        root="work/data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.MNIST(
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
    ITER = 10
    epoch = 0
    for _ in range(ITER):
        print(f"--------------------")
        print(f"Epoch {epoch + 1}")
        print(f"--------------------")
        train_loss, train_acc = train(
            train_dataloader, device, model, loss_fn, optimizer
        )
        test(test_dataloader, device, model, loss_fn)
        epoch += 1

    # Save a checkpoint 
    # Information for training can be saved in dict
    checkpoint_path = f"work/nn_epoch_{epoch}.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss
        },
        checkpoint_path
    )
    print(f'Save a checkpoint to "{checkpoint_path}"')

    print(f"----------------------------------------")
    print(f'Load the checkpoint from "{checkpoint_path}"')
    print(f"----------------------------------------")
    cpt = torch.load(checkpoint_path)

    # Load the model and optimizer
    model = SimpleNN().to(device)
    model.load_state_dict(cpt["model_state_dict"])

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(cpt["optimizer_state_dict"])

    # Resume training
    epoch = cpt["epoch"]
    for _ in range(ITER):
        print(f"--------------------")
        print(f"Epoch {epoch + 1}")
        print(f"--------------------")
        train_loss, train_acc = train(
            train_dataloader, device, model, loss_fn, optimizer
        )
        test(test_dataloader, device, model, loss_fn)
        epoch += 1

    # Save a checkpoint again
    checkpoint_path = f"work/nn_epoch_{epoch}.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss
        },
        checkpoint_path
    )
    print(f'Save a checkpoint to "{checkpoint_path}"')

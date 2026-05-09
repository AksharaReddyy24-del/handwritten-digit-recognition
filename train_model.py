from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = Path(__file__).resolve().parent / "mnist_cnn_model.pth"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses: List[float] = []
    accuracies: List[float] = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Loss: {epoch_loss:.4f}, "
            f"Test Accuracy: {accuracy:.4f}"
        )

    return losses, accuracies


def main():
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()

    print("Building CNN model...")
    model = CNN()

    print("Training model...")
    losses, accuracies = train_and_evaluate(model, train_loader, test_loader)

    print(f"Final Test Accuracy: {accuracies[-1] * 100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
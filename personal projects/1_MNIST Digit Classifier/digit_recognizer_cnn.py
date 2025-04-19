######################################################################
# Set Up the Environment
######################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd

# Create a directory to save model and plots
os.makedirs("output", exist_ok=True)

######################################################################
# Load and Preprocess the MNIST Dataset
######################################################################
# Define transformations
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Download dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

# Apply transformations
train_dataset.transform = train_transform
test_dataset.transform = test_transform

# Split training data for validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


######################################################################
# Visualize Sample Data
######################################################################
def visualize_samples(dataset):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        img, label = dataset[i]
        img = img.squeeze().numpy()
        plt.subplot(5, 5, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.suptitle("Sample Training Images", fontsize=16)
    plt.savefig("output/sample_training_images.png")
    plt.show()


visualize_samples(train_dataset)


######################################################################
# Define an Advanced Neural Network Model (CNN-based)
######################################################################
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


######################################################################
# Train the Model
######################################################################
def train_model(model, train_loader, val_loader, epochs=10):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, counter = 2, 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "output/mnist_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses


train_losses, val_losses = train_model(model, train_loader, val_loader)

######################################################################
# Plot Training History
######################################################################
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/training_history.png")
plt.show()


######################################################################
# Evaluate the Model
######################################################################
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load("output/mnist_model.pth"))
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return all_labels, all_preds


labels, preds = evaluate_model(model, test_loader)

######################################################################
# Confusion Matrix
######################################################################
conf_matrix = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("output/confusion_matrix.png")
plt.show()

######################################################################
# Classification Report
######################################################################
print("Classification Report:")
print(classification_report(labels, preds))

######################################################################
# Save Predictions to CSV
######################################################################
results = pd.DataFrame({"True Label": labels, "Predicted Label": preds})
results.to_csv("output/predictions.csv", index=False)
print("Predictions exported to output/predictions.csv")


######################################################################
# Robust Error Analysis: Misclassified Images and F1-score per Class
######################################################################
def analyze_errors(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            misclassified.extend(
                [
                    (img, pred, lbl)
                    for img, pred, lbl in zip(
                        images.cpu(), predicted.cpu(), labels.cpu()
                    )
                    if pred != lbl
                ]
            )

    # Compute F1-score per class
    f1_scores = f1_score(all_labels, all_preds, average=None)
    print("F1-scores per class:", f1_scores)

    # Visualize misclassified images
    plt.figure(figsize=(10, 10))
    for i, (img, pred, lbl) in enumerate(misclassified[:25]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Pred: {pred}, True: {lbl}")
        plt.axis("off")
    plt.suptitle("Misclassified Examples", fontsize=16)
    plt.savefig("output/misclassified_examples.png")
    plt.show()

    return all_labels, all_preds


labels, preds = analyze_errors(model, test_loader)

######################################################################
# Save Predictions to CSV
######################################################################
results = pd.DataFrame({"True Label": labels, "Predicted Label": preds})
results.to_csv("output/predictions.csv", index=False)
print("Predictions exported to output/predictions.csv")

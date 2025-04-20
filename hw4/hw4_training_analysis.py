import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.models import SimpleNN, DeepNN, CNN1, CNN2
from utils.data_utils import load_data
from utils.train_utils import train_model, plot_curves, plot_conf_matrix

def load_data(data_path):
    with open(data_path, 'rb') as f:
        train_images, test_images, train_labels, test_labels = pickle.load(f)

    X_train_flat = torch.tensor(train_images / 255.0, dtype=torch.float32)
    X_test_flat = torch.tensor(test_images / 255.0, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    X_train_cnn = X_train_flat.view(-1, 1, 28, 28)
    X_test_cnn = X_test_flat.view(-1, 1, 28, 28)

    return X_train_flat, y_train, X_test_flat, y_test, X_train_cnn, X_test_cnn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 40, kernel_size=6)
        self.relu = nn.ReLU()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.relu(self.conv(dummy))
            self.flat_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flat_size, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.relu1(self.conv(dummy))
            self.flat_size = out.view(1, -1).size(1)
        self.fc1 = nn.Linear(self.flat_size, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_loss, correct, total = 0, 0, 0
            for X_batch, y_batch in train_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                correct += (output.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

            train_losses.append(total_loss / total)
            train_accuracies.append(correct / total)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return train_losses, train_accuracies, test_acc, f1, all_preds, all_labels

def plot_curves(losses, accs, model_name, fig_dir, epoch_tag):
    epochs = list(range(1, len(losses)+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, losses, marker='o')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')

    plt.subplot(1,2,2)
    plt.plot(epochs, accs, marker='o')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{model_name.lower().replace(" ", "_")}_curves_{epoch_tag}.png'))
    plt.close()

def plot_conf_matrix(labels, preds, model_name, fig_dir, epoch_tag):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')  # <- values_format='d' ensures integer display
    plt.title(f"Confusion Matrix - {model_name} ({epoch_tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{model_name.lower().replace(" ", "_")}_confusion_{epoch_tag}.png'))
    plt.close()

import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
import matplotlib.pyplot as plt
from utils.data_utils import load_data
from torch.utils.data import DataLoader, TensorDataset
from utils.train_utils import train_model, evaluate_model
from models.models import SimpleNN, DeepNN, CNN1, CNN2, EnhancedCNN2

# -------------------------------
# Load and Prepare Data
# -------------------------------
data_path = '/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw4/data/fashion_mnist.p'

X_train_flat, y_train, X_test_flat, y_test, X_train_cnn, X_test_cnn = load_data(data_path)

# Use MB Pro M3 par pool of 14 workers
train_loader = DataLoader(TensorDataset(X_train_flat, y_train), batch_size=64, shuffle=True, num_workers=14)
test_loader = DataLoader(TensorDataset(X_test_flat, y_test), batch_size=1000, num_workers=14)
cnn_train_loader = DataLoader(TensorDataset(X_train_cnn, y_train), batch_size=64, shuffle=True, num_workers=14)
cnn_test_loader = DataLoader(TensorDataset(X_test_cnn, y_test), batch_size=1000, num_workers=14)

# Normalize and flatten for fully connected nets
#X_train_flat = torch.tensor(train_images / 255.0, dtype=torch.float32)
#X_test_flat = torch.tensor(test_images / 255.0, dtype=torch.float32)
#y_train = torch.tensor(train_labels, dtype=torch.long)
#y_test = torch.tensor(test_labels, dtype=torch.long)

# Use MB Pro M3 par pool of 14 workers
#train_loader = DataLoader(TensorDataset(X_train_flat, y_train), batch_size=64, shuffle=True, num_workers=14)
#test_loader = DataLoader(TensorDataset(X_test_flat, y_test), batch_size=1000, num_workers=14)

# train_loader = DataLoader(TensorDataset(X_train_flat, y_train), batch_size=64, shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test_flat, y_test), batch_size=1000)

# Also reshape for CNNs: (N, 1, 28, 28)
#X_train_cnn = X_train_flat.view(-1, 1, 28, 28)
#X_test_cnn = X_test_flat.view(-1, 1, 28, 28)

# Use MB Pro M3 par pool of 14 workers
#cnn_train_loader = DataLoader(TensorDataset(X_train_cnn, y_train), batch_size=64, shuffle=True, num_workers=14)
#cnn_test_loader = DataLoader(TensorDataset(X_test_cnn, y_test), batch_size=1000, num_workers=14)

# cnn_train_loader = DataLoader(TensorDataset(X_train_cnn, y_train), batch_size=64, shuffle=True)
# cnn_test_loader = DataLoader(TensorDataset(X_test_cnn, y_test), batch_size=1000)

# -------------------------------
# Model 1: Shallow NN
# -------------------------------
# class SimpleNN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.fc1 = nn.Linear(784, 128)
#        self.relu = nn.ReLU()
#        self.fc2 = nn.Linear(128, 10)
#
#    def forward(self, x):
#        x = self.relu(self.fc1(x))
#        return self.fc2(x)

# -------------------------------
# Model 2: Deeper NN
# -------------------------------
#class DeepNN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.fc1 = nn.Linear(784, 128)
#        self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(128, 32)
#        self.relu2 = nn.ReLU()
#        self.fc3 = nn.Linear(32, 10)
#
#    def forward(self, x):
#        x = self.relu1(self.fc1(x))
#        x = self.relu2(self.fc2(x))
#        return self.fc3(x)

# -------------------------------
# Training for Fully Connected Networks
# -------------------------------
def train_model(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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
        train_loss = total_loss / total
        train_acc = correct / total

        correct, total = 0, 0
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            correct += (output.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
        test_acc = correct / total

    return train_loss, train_acc, test_acc

# -------------------------------
# 6.2(a): CNN Model
# -------------------------------
#class CNN1(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv = nn.Conv2d(1, 40, kernel_size=6)
#        self.relu = nn.ReLU()
#        with torch.no_grad():
#            dummy = torch.zeros(1, 1, 28, 28)
#            out = self.relu(self.conv(dummy))
#            self.flat_size = out.view(1, -1).size(1)
#        self.fc = nn.Linear(self.flat_size, 10)
#
#    def forward(self, x):
#        x = self.relu(self.conv(x))
#        x = x.view(x.size(0), -1)
#        return self.fc(x)

def train_eval(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss, train_correct, train_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            out = model(X_batch)
            loss = criterion(out, y_batch)
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (out.argmax(dim=1) == y_batch).sum().item()
            train_total += y_batch.size(0)

        test_correct, test_total = 0, 0
        for X_batch, y_batch in test_loader:
            out = model(X_batch)
            test_correct += (out.argmax(dim=1) == y_batch).sum().item()
            test_total += y_batch.size(0)

    return train_loss / train_total, train_correct / train_total, test_correct / test_total

# -------------------------------
# 6.2(b): Deeper CNN Model
# -------------------------------
#class CNN2(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv = nn.Conv2d(1, 32, kernel_size=5)
#        self.relu1 = nn.ReLU()
#        
#        # Determine flattened size dynamically
#        with torch.no_grad():
#            dummy = torch.zeros(1, 1, 28, 28)
#            out = self.relu1(self.conv(dummy))
#            self.flat_size = out.view(1, -1).size(1)
#
#        self.fc1 = nn.Linear(self.flat_size, 32)
#        self.relu2 = nn.ReLU()
#        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Train fully connected models
    model1 = SimpleNN()
    model2 = DeepNN()
    train_loss1, train_acc1, test_acc1 = train_model(model1, train_loader, test_loader)
    train_loss2, train_acc2, test_acc2 = train_model(model2, train_loader, test_loader)

    print("Model 1 (784 → 128 → 10)")
    print(f"Train CE Loss: {train_loss1:.4f} | Train Acc: {train_acc1:.4f} | Test Acc: {test_acc1:.4f}\n")

    print("Model 2 (784 → 128 → 32 → 10)")
    print(f"Train CE Loss: {train_loss2:.4f} | Train Acc: {train_acc2:.4f} | Test Acc: {test_acc2:.4f}\n")

    # CNN Models
    model_cnn1 = CNN1()
    model_cnn2 = CNN2()
    train_loss_c1, train_acc_c1, test_acc_c1 = train_model(model_cnn1, cnn_train_loader, cnn_test_loader)
    train_loss_c2, train_acc_c2, test_acc_c2 = train_model(model_cnn2, cnn_train_loader, cnn_test_loader)

    print("CNN 6.2(a): Conv(1→40, 6x6) → ReLU → Linear(…) → 10")
    print(f"Train CE Loss: {train_loss_c1:.4f} | Train Acc: {train_acc_c1:.4f} | Test Acc: {test_acc_c1:.4f}")

    print("CNN 6.2(b): Conv(1→32, 5x5) → ReLU → Linear(…) → ReLU → Linear(10)")
    print(f"Train CE Loss: {train_loss_c2:.4f} | Train Acc: {train_acc_c2:.4f} | Test Acc: {test_acc_c2:.4f}")

    # Extra Models
    model_cnn3 = EnhancedCNN2()
    losses_c3, accs_c3, test_acc_c3 = train_model(model_cnn3, cnn_train_loader, cnn_test_loader)
    # losses_c3, accs_c3, test_acc_c3, f1_c3, preds_c3, labels_c3 = train_model(model_cnn3, cnn_train_loader, cnn_test_loader)

    print("CNN3: Extra Model 1")
    print(f"Train CE Loss: {losses_c3:.4f} | Train Acc: {accs_c3:.4f} | Test Acc: {test_acc_c3:.4f}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
import torch
import torch.nn as nn

# ------------------------------------
# Model 1: Shallow Fully Connected Network
# Architecture: 784 -> 128 -> 10
# ------------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------------
# Model 2: Deeper Fully Connected Network
# Architecture: 784 -> 128 -> 32 -> 10
# ------------------------------------
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

# ------------------------------------
# Model 3: Shallow CNN
# Architecture: Conv(1->40, 6x6) -> ReLU -> Flatten -> Linear(10)
# ------------------------------------
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 40, kernel_size=6)
        self.relu = nn.ReLU()

        # Dynamically compute the flattened size after conv
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.relu(self.conv(dummy))
            self.flat_size = out.view(1, -1).size(1)

        self.fc = nn.Linear(self.flat_size, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------------------
# Model 4: Deeper CNN
# Architecture: Conv(1->32, 5x5) -> ReLU -> Linear(…) -> ReLU -> Linear(10)
# ------------------------------------
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()

        # Dynamically compute the flattened size after conv
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
        return self.fc2(x)
    
# ------------------------------------
# Extra Model 1: CNN3 Batch Norm
# Architecture: Conv(1→32, 5x5)+Conv(32→64, 3x3)+BN+Dropout → MaxPool → Linear(128) → Linear(10)
# ------------------------------------
class EnhancedCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.25)

        self.pool = nn.MaxPool2d(2, 2)  # reduces 28x28 → 14x14
        self.flat_size = 64 * 14 * 14

        self.fc1 = nn.Linear(self.flat_size, 128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop3(self.relu3(self.fc1(x)))
        return self.fc2(x)

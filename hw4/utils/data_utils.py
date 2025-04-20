import pickle
import torch

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
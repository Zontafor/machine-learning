import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

def train_model(model, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate training performance for this epoch
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

    # Final evaluation on test set
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_labels': all_labels,
        'predictions': all_preds
    }
    
def evaluate_model(model, test_loader):
    """
    Evaluates a trained model on the test set.
    Returns: test accuracy, macro F1-score, confusion matrix, true labels, predictions
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return acc, f1, cm, all_labels, all_preds

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name} ({epoch_tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{model_name.lower().replace(" ", "_")}_confusion_{epoch_tag}.png'))
    plt.close()
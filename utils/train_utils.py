import os
import matplotlib.pyplot as plt

try:
    import torch
except ModuleNotFoundError:
    torch = None


def train_one_epoch(model, loader, criterion, optimizer, device):
    if torch is None:
        raise ImportError("PyTorch is required to train CNN and ResNet models.")

    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, y_true, y_pred


def evaluate(model, loader, criterion, device):
    if torch is None:
        raise ImportError("PyTorch is required to evaluate CNN and ResNet models.")

    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, y_true, y_pred


def save_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_accuracy_curve(train_accs, val_accs, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        improved = val_loss < self.best_loss
        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

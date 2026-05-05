import argparse
import os
import random

import numpy as np

from models.mlp import MLPClassifier
from utils.metrics import compute_classification_metrics
from utils.numpy_dataset import get_numpy_dataloaders
from utils.train_utils import (
    train_one_epoch,
    evaluate,
    save_loss_curve,
    save_accuracy_curve,
    save_confusion_matrix,
    EarlyStopping,
    ensure_dir,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from models.cnn import SimpleCNN
    from models.resnet import get_resnet18
    from utils.dataset import get_dataloaders
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None
    SimpleCNN = None
    get_resnet18 = None
    get_dataloaders = None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch and torchvision are required for cnn/resnet18. "
            "Use --model mlp for the NumPy-only implementation."
        )


def get_model(model_name, num_classes):
    if model_name == "mlp":
        return MLPClassifier(num_classes)
    require_torch()
    if model_name == "cnn":
        return SimpleCNN(num_classes)
    if model_name == "resnet18":
        return get_resnet18(num_classes, pretrained=True)
    raise ValueError("model_name must be 'mlp', 'cnn', or 'resnet18'")


def as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def train_one_epoch_numpy(model, loader):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images_np = as_numpy(images)
        labels_np = as_numpy(labels).astype(np.int64)

        loss, preds = model.train_batch(images_np, labels_np)
        running_loss += loss * images_np.shape[0]

        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, y_true, y_pred


def evaluate_numpy(model, loader):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images_np = as_numpy(images)
        labels_np = as_numpy(labels).astype(np.int64)

        loss, preds = model.loss_and_predictions(images_np, labels_np)
        running_loss += loss * images_np.shape[0]

        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, y_true, y_pred


def run_numpy_mlp(args, train_loader, val_loader, test_loader, num_classes, save_dir):
    model = MLPClassifier(
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    early_stopper = EarlyStopping(patience=args.patience)
    best_model_path = os.path.join(save_dir, "best_model.npz")

    for epoch in range(args.epochs):
        train_loss, train_true, train_pred = train_one_epoch_numpy(model, train_loader)
        train_acc, train_f1, _ = compute_classification_metrics(train_true, train_pred)

        val_loss, val_true, val_pred = evaluate_numpy(model, val_loader)
        val_acc, val_f1, _ = compute_classification_metrics(val_true, val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        if early_stopper.step(val_loss):
            model.save(best_model_path)

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

    model.load(best_model_path)

    test_loss, test_true, test_pred = evaluate_numpy(model, test_loader)
    test_acc, test_f1, cm = compute_classification_metrics(test_true, test_pred)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")

    save_loss_curve(train_losses, val_losses, os.path.join(save_dir, "loss_curve.png"))
    save_accuracy_curve(train_accs, val_accs, os.path.join(save_dir, "accuracy_curve.png"))
    save_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"))

    with open(os.path.join(save_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write("Implementation: NumPy MLP, no PyTorch/TensorFlow layers or optimizer\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n")


def main(args):
    set_seed(args.seed)

    save_dir = os.path.join("outputs", f"{args.dataset}_{args.model}")
    ensure_dir(save_dir)

    if args.model == "mlp":
        train_loader, val_loader, test_loader, num_classes = get_numpy_dataloaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            use_augmentation=not args.no_augmentation,
            seed=args.seed,
        )
        run_numpy_mlp(args, train_loader, val_loader, test_loader, num_classes, save_dir)
        return

    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation,
        seed=args.seed,
    )

    model = get_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    early_stopper = EarlyStopping(patience=args.patience)
    best_model_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(args.epochs):
        train_loss, train_true, train_pred = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_acc, train_f1, _ = compute_classification_metrics(train_true, train_pred)

        val_loss, val_true, val_pred = evaluate(
            model, val_loader, criterion, device
        )
        val_acc, val_f1, _ = compute_classification_metrics(val_true, val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        if early_stopper.step(val_loss):
            torch.save(model.state_dict(), best_model_path)

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_acc, test_f1, cm = compute_classification_metrics(test_true, test_pred)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")

    save_loss_curve(train_losses, val_losses, os.path.join(save_dir, "loss_curve.png"))
    save_accuracy_curve(train_accs, val_accs, os.path.join(save_dir, "accuracy_curve.png"))
    save_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"))

    with open(os.path.join(save_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "resnet18"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_augmentation", action="store_true")
    args = parser.parse_args()

    main(args)

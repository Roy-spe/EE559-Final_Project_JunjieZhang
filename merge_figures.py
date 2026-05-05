import os
import matplotlib.pyplot as plt
import argparse

def merge_curves(dataset, curve_type):
    """
    Merge accuracy or loss curves for all three models into one figure.
    """
    models = ['mlp', 'cnn', 'resnet18']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, model in enumerate(models):
        path = os.path.join('outputs', f'{dataset}_{model}', f'{curve_type}_curve.png')
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {model}.")
            continue
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(f'{model.upper()} {curve_type.capitalize()} Curve')
        axes[i].axis('off')

    plt.tight_layout()
    output_path = os.path.join('outputs', f'{dataset}_merged_{curve_type}_curves.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved merged {curve_type} curves to {output_path}")

def merge_confusion_matrices(dataset):
    """
    Merge confusion matrices for all three models into one figure.
    """
    models = ['mlp', 'cnn', 'resnet18']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, model in enumerate(models):
        path = os.path.join('outputs', f'{dataset}_{model}', 'confusion_matrix.png')
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {model}.")
            continue
        img = plt.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(f'{model.upper()} Confusion Matrix')
        axes[i].axis('off')

    plt.tight_layout()
    output_path = os.path.join('outputs', f'{dataset}_merged_confusion_matrices.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved merged confusion matrices to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge figures for CIFAR-10 and CIFAR-100 models.")
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True,
                        help="Dataset to merge figures for.")
    args = parser.parse_args()

    dataset = args.dataset

    # Merge accuracy curves
    merge_curves(dataset, 'accuracy')

    # Merge loss curves
    merge_curves(dataset, 'loss')

    # Merge confusion matrices
    merge_confusion_matrices(dataset)

if __name__ == "__main__":
    main()

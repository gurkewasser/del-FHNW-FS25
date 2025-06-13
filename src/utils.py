import matplotlib.pyplot as plt
import hashlib
import os
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, run_number, model_name, learning_rate, batch_size):
    
    fig = plt.figure(figsize=(12, 4))
    title = f"Run {run_number} - {model_name}, lr={learning_rate}, bs={batch_size}"
    fig.suptitle(title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Verlauf der Kostenfunktion")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Verlauf der Genauigkeit")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_cv_cache_name(model_class, dataset, num_classes, k, epochs, batch_size, lr):
    """
    Create a unique cache filename for cross-validation results, using as much dataset-specific
    information as possible. This version hashes the indices of the first 100 samples and the class
    distribution, so that different datasets (even with the same root) get different cache files.
    """
    # Try to get a unique identifier for the dataset
    dataset_id_parts = []

    # Use root path if available
    if hasattr(dataset, 'root'):
        dataset_id_parts.append(str(dataset.root))
    # Use transforms if available (to distinguish resize/crop)
    if hasattr(dataset, 'transform') and dataset.transform is not None:
        dataset_id_parts.append(str(dataset.transform))
    # Use class names if available
    if hasattr(dataset, 'classes'):
        dataset_id_parts.append(str(dataset.classes))
    # Use targets if available (for class distribution)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, list):
            targets_arr = np.array(targets)
        else:
            targets_arr = np.array(targets)
        class_hist = np.bincount(targets_arr) if len(targets_arr) > 0 else []
        dataset_id_parts.append("hist" + str(class_hist.tolist()))
        # Hash the first 100 indices to distinguish different splits
        first_100 = targets_arr[:100]
        dataset_id_parts.append("first100_" + hashlib.md5(first_100.tobytes()).hexdigest())
    else:
        # Fallback: use length
        dataset_id_parts.append("len" + str(len(dataset)))

    dataset_id = "_".join(dataset_id_parts)

    # Model class name
    model_name = model_class.__name__
    # Compose a string with all params
    cache_str = f"{model_name}_{dataset_id}_nc{num_classes}_k{k}_ep{epochs}_bs{batch_size}_lr{lr}"
    # Hash to avoid long filenames
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
    cache_dir = "./cv_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cv_{cache_hash}.pkl")
    return cache_path

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total
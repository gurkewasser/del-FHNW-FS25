import matplotlib.pyplot as plt

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
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from dataloader import get_dataloaders
from model import get_model
import wandb

# Configurable Parameters
config = {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "image_size": 224
}

wandb.init(project="del-FS24", config=config)

def train(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Track loss over epochs
    history = {
    "epoch": [],
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": []
}

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Compute validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Berechne Training Accuracy
        model.eval()
        train_correct, train_total = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total

        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        # Save loss values
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save training history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv("logs/training_history.csv", index=False)
    print("Training history saved to logs/training_history.csv")

    # Save trained model
    model_path = "logs/model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Log model to wandb
    wandb.save(model_path)

def overfit_on_one_batch(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Nur 1 Mini-Batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    for epoch in range(50):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean().item() * 100
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders(batch_size=config["batch_size"], image_size=config["image_size"])
    
    model = get_model(model_type="simple")
    #overfit_on_one_batch(model, train_loader) # overfit tester
    train(model, train_loader, val_loader, config)
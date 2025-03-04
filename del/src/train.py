import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from dataloader import get_dataloaders
from model import get_model

# Configurable Parameters
config = {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
}

def train(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {"epoch": [], "train_loss": [], "val_loss": []}  # Store loss values

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

        # Compute average training loss
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

        # Save results
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save history to CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv("logs/training_history.csv", index=False)
    print("Training history saved to training_history.csv")

    # Save trained model
    torch.save(model.state_dict(), "logs/model.pth")

if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders(batch_size=config["batch_size"])
    model = get_model()
    train(model, train_loader, val_loader, config)
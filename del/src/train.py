import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_dataloaders
from model import get_model

def train(model, train_loader, val_loader, num_epochs=10, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()  # Fix: Expect three values
    model = get_model()
    train(model, train_loader, val_loader)
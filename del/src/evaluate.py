import torch
from model import get_model
from dataloader import get_dataloaders

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:  # Use test_loader instead of val_loader
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    _, _, test_loader = get_dataloaders()  # Fix: Unpack three values properly
    model = get_model()
    evaluate(model, test_loader)
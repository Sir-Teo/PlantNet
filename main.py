import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_dataset import load_dataset
from net import SqueezeNet, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(val_loader), accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_dataset(args.data_dir, args.batch_size)

    if args.model == "squeezenet":
        model = SqueezeNet(num_classes=args.num_classes).to(device)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=args.num_classes).to(device)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_classes=args.num_classes).to(device)
    elif args.model == "efficientnetb0":
        model = EfficientNetB0(num_classes=args.num_classes).to(device)
    elif args.model == "inceptionv3":
        model = InceptionV3(num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Plant Seedlings Classification")
    parser.add_argument("--model", type=str, default="squeezenet", help="Model to use (squeezenet, resnet50, mobilenetv2, efficientnetb0, inceptionv3)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_classes", type=int, default=12, help="Number of classes in the dataset")
    args = parser.parse_args()
    main(args)
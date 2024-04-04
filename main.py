import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_dataset import load_dataset, split_dataset
from net import SqueezeNet, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += images.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / total_samples
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device, epoch, writer):
    print(f"Test loader length: {len(test_loader)}")
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = running_corrects.double() / total_samples
    epoch_precision = precision_score(true_labels, pred_labels, average='weighted')
    epoch_recall = recall_score(true_labels, pred_labels, average='weighted')
    epoch_f1 = f1_score(true_labels, pred_labels, average='weighted')

    writer.add_scalar('Loss/test', epoch_loss, epoch)
    writer.add_scalar('Accuracy/test', epoch_acc, epoch)
    writer.add_scalar('Precision/test', epoch_precision, epoch)
    writer.add_scalar('Recall/test', epoch_recall, epoch)
    writer.add_scalar('F1-score/test', epoch_f1, epoch)

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = load_dataset(args.data_dir)

    # Split the dataset into train, test, and validation sets
    train_dataset, test_dataset, val_dataset = split_dataset(dataset, args.test_size, args.val_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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

    run_name = f"{args.model}_lr{args.learning_rate}_bs{args.batch_size}_epochs{args.num_epochs}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device, epoch, writer)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

    plot_metrics(train_losses, train_accs, val_losses, val_accs, val_precisions, val_recalls, val_f1s, run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    writer.flush()
    writer.close()

def plot_metrics(train_losses, train_accs, test_losses, test_accs, test_precisions, test_recalls, test_f1s, run_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, test_losses, 'r', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'loss_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, test_precisions, 'b', label='Test Precision')
    plt.plot(epochs, test_recalls, 'r', label='Test Recall')
    plt.plot(epochs, test_f1s, 'g', label='Test F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'precision_recall_f1.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Plant Seedlings Classification")
    parser.add_argument("--model", type=str, default="squeezenet", help="Model to use (squeezenet, resnet50, mobilenetv2, efficientnetb0, inceptionv3)")
    parser.add_argument("--data_dir", type=str, default="data/train/", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_classes", type=int, default=12, help="Number of classes in the dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of the dataset to be used as test set")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of the dataset to be used as validation set")
    args = parser.parse_args()
    main(args)
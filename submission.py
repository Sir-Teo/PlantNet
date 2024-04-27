# submission.py
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from net import SqueezeNet, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3
import csv

from load_dataset import PlantSeedlingsDataset

def load_model(model_name, num_classes, checkpoint_path):
    if model_name == "squeezenet":
        model = SqueezeNet(num_classes=num_classes)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif model_name == "mobilenetv2":
        model = MobileNetV2(num_classes=num_classes)
    elif model_name == "efficientnetb0":
        model = EfficientNetB0(num_classes=num_classes)
    elif model_name == "inceptionv3":
        model = InceptionV3(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path, transform, classes):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def main(args):
    checkpoint_path = os.path.join("results", args.run_name, "best_model.pth")
    model = load_model(args.model, args.num_classes, checkpoint_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the classes from the dataset
    dataset = PlantSeedlingsDataset(args.data_dir+"/train", transform=None)
    classes = dataset.classes
    
    test_dir = os.path.join(args.data_dir, "test")
    submission_file = "submission.csv"
    with open(submission_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "species"])
        for filename in os.listdir(test_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(test_dir, filename)
                predicted_class = predict(model, image_path, transform, classes)
                writer.writerow([filename, predicted_class])
    
    print(f"Submission file '{submission_file}' generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission CSV")
    parser.add_argument("--model", type=str, required=True, help="Model to use (squeezenet, resnet50, mobilenetv2, efficientnetb0, inceptionv3)")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run directory containing the trained model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--num_classes", type=int, default=12, help="Number of classes in the dataset")
    args = parser.parse_args()
    main(args)
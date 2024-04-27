import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image

class PlantSeedlingsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = sorted(entry.name for entry in os.scandir(self.data_dir) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.data_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.endswith(".png"):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        samples.append(item)
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

def load_dataset(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the same fixed size as train_transform
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PlantSeedlingsDataset(data_dir)
    return dataset, train_transform, test_transform

def split_dataset(dataset, test_size, val_size, train_transform, test_transform):
    test_size = int(len(dataset) * test_size)
    if val_size > 0:
        val_size = int(len(dataset) * val_size)
        train_size = len(dataset) - test_size - val_size
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size, val_size]
        )
        val_dataset.dataset.transform = test_transform
    else:
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        val_dataset = None

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    return train_dataset, test_dataset, val_dataset
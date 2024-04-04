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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PlantSeedlingsDataset(data_dir, transform=transform)
    return dataset

def split_dataset(dataset, test_size, val_size):
    test_size = int(test_size * len(dataset))
    val_size = int(val_size * len(dataset))
    train_size = len(dataset) - test_size - val_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    return train_dataset, test_dataset, val_dataset
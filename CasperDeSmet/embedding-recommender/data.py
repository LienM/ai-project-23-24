import os
from pathlib import Path

from PIL import Image
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# Adapted from https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/3
class UnlabeledImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = [(directory, Path(filename)) for (directory, _, filenames) in os.walk(root) for filename in filenames]
        self.images = [(directory, filename) for (directory, filename) in self.images if filename.suffix == ".jpg"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        directory, image_name = self.images[index]
        image_path = os.path.join(directory, image_name)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_name.stem

def load_data(directory, batch_size):
    transform = models.VGG19_BN_Weights.IMAGENET1K_V1.transforms()
    dataset = UnlabeledImageDataset(directory, transform)
    return DataLoader(dataset, batch_size)
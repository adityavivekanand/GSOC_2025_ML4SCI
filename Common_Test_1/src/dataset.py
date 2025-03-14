import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith('.npy'):
                    self.samples.append((os.path.join(class_path, fname), idx))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = np.load(path)  # Load the image from .npy file
        if image.ndim == 3 and image.shape[-1] in [1, 3]:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if self.transform:
            image = self.transform(image)
        return image, label

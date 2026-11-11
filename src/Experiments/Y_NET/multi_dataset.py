import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class SpineDataset(Dataset):
    def __init__(self, split="train", root_dir="../data/processed", transform=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, split, "images", "*.npy"))
        )
        self.mask_paths = sorted(
            glob.glob(os.path.join(root_dir, split, "masks", "*.npy"))
        )

        assert len(self.image_paths) == len(self.mask_paths), (
            "Mismatch between images and masks!"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        if self.transform:
            image, mask = self._augment(image, mask)

        image = torch.from_numpy(image[np.newaxis, ...].copy()).float()
        mask = torch.from_numpy(mask).long()
        return image, mask

    def _augment(self, image, mask):

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if np.random.rand() > 0.5:
            image = np.clip(image + np.random.uniform(-0.1, 0.1), 0, 1)

        return image, mask

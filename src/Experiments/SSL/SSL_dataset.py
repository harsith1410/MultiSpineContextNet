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


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = np.load(self.image_paths[idx])

        if self.transform:
            image = self._augment(image)

        masked_image = self.apply_block_mask(image)

        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if masked_image.ndim == 3:
            masked_image = masked_image[np.newaxis, ...]



        image = torch.from_numpy(image).float()
        masked_image = torch.from_numpy(masked_image).float()

        return masked_image, image

    def _augment(self, image):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0).copy()


        if np.random.rand() > 0.5:
            image = np.clip(image + np.random.uniform(-0.1, 0.1), 0, 1)

        return image

    def apply_block_mask(self, image, num_blocks=256, block_size=(16, 16, 8)):
        masked_image = image.copy()

        d, h, w = masked_image.shape
        b_d, b_h, b_w = block_size

        for _ in range(num_blocks):
            z_start = np.random.randint(0, d - b_d + 1)
            y_start = np.random.randint(0, h - b_h + 1)
            x_start = np.random.randint(0, w - b_w + 1)

            masked_image[z_start:z_start + b_d,
            y_start:y_start + b_h,
            x_start:x_start + b_w] = 0.0

        return masked_image

from pathlib import Path

import torch
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
import torch.nn.functional as F
from config import PATCHES_TEMPLATE, CROPPED_TEMPLATE, PATCH_SIZE, STRIDE

from loguru import logger


class Data(torch.utils.data.Dataset):
    def __init__(
            self, data_dir: str, n_images: int,
    ):
        self.ground_truth = []
        self.input_images = []
        ground_truth_dir = Path(data_dir) / PATCHES_TEMPLATE.format(PATCH_SIZE)
        cropped_dir = Path(data_dir) / CROPPED_TEMPLATE.format(PATCH_SIZE, STRIDE)

        if not cropped_dir.exists() and not ground_truth_dir.exists():
            raise FileNotFoundError(
                "Make sure you've preprocessed your imagery with this configuration:"
                f"\n{STRIDE = } \n{PATCH_SIZE = }"
            )
        ground_truth_files = sorted(list(ground_truth_dir.glob("*.jpg")))  # dangerous wildcards
        cropped_files = sorted(list(cropped_dir.glob("*.jpg")))

        if not ground_truth_files or not cropped_files:
            raise FileNotFoundError("Empty dirs")

        n_images = len(ground_truth_files) if n_images == -1 else n_images

        for index in tqdm(range(n_images), desc="loading images"):
            patch = np.array(io.imread(ground_truth_files[index]))
            patch = patch.transpose(2, 0, 1) / 255
            self.ground_truth.append(torch.tensor(patch, dtype=torch.float32))

            cropped_image = np.array(io.imread(cropped_files[index]))
            # cropped_image = cropped_image[margin_size:-margin_size, margin_size:-margin_size]
            cropped_image = cropped_image.transpose(2, 0, 1) / 255
            cropped_image = torch.tensor(cropped_image, dtype=torch.float32)
            # todo make this dependent on the patch_size:
            cropped_image = F.pad(cropped_image, (20, 20, 20, 20), mode='reflect')
            self.input_images.append(torch.tensor(cropped_image, dtype=torch.float32))

        self.input_images = torch.stack(self.input_images)
        self.ground_truth = torch.stack(self.ground_truth)

    def __getitem__(self, index):
        return self.input_images[index], self.ground_truth[index]

    def __len__(self):
        return len(self.input_images)

    def data_train_test_split(self, test_size_percent: float = .2, shuffle: bool = True):
        """ not returning dataloaders"""
        train_size = len(self) - int(test_size_percent * len(self))
        random_indices = np.random.randint(0, len(self), len(self))
        train_mask, test_mask = random_indices[:train_size], random_indices[train_size:]
        x_train, y_train = self[train_mask]
        x_test, y_test = self[test_mask]
        return x_train, y_train, x_test, y_test


def compare_images(image_a, image_b):
    """ could be the ground truth and the outputs of a UNet or a baseline"""
    # metrics
    mean_squared = mean_squared_error(image_a, image_b)
    # ssim = structural_similarity(image, baseline_image.squeeze())  # needs to be grayscale
    psnr = peak_signal_noise_ratio(image_a, image_b)
    logger.info(f"{mean_squared=}, {psnr=}")
    plot = False
    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(image_a[:, :, 0], cmap='jet')
        ax[0, 1].imshow(image_b[:, :, 0], cmap='jet')
        ax[1, 0].imshow(image_a)
        ax[1, 1].imshow(image_b)
        fig.suptitle(f"{STRIDE=} {PATCH_SIZE=}\n MSE: {mean_squared:.2f}, PSNR {psnr:.2f}dB")

    return fig
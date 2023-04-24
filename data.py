from pathlib import Path

import torch
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import PATCHES_TEMPLATE, CROPPED_TEMPLATE, PATCH_SIZE, STRIDE


class Data(torch.utils.data.Dataset):
    def __init__(
            self, data_dir: str, n_images: int, margin_size: int = 20
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
        ground_truth_files = list(ground_truth_dir.glob("*.jpg"))
        cropped_files = list(cropped_dir.glob("*.jpg"))

        if not ground_truth_files or not cropped_files:
            raise FileNotFoundError("Empty dirs")

        n_images = len(ground_truth_files) if n_images == -1 else n_images

        for index in tqdm(range(n_images), desc="loading images"):
            patch = np.array(io.imread(ground_truth_files[index]))
            patch = patch.transpose(2, 0, 1) / 255
            self.ground_truth.append(torch.tensor(patch, dtype=torch.float32))

            cropped_image = np.array(io.imread(cropped_files[index]))
            cropped_image = cropped_image[margin_size:-margin_size, margin_size:-margin_size]
            cropped_image = cropped_image.transpose(2, 0, 1) / 255
            self.input_images.append(torch.tensor(cropped_image, dtype=torch.float32))

    def __getitem__(self, index):
        return self.ground_truth[index], self.input_images[index]

    def __len__(self):
        return len(self.input_images)

    def data_train_test_split(self, test_size_percent: float = .2, shuffle: bool = True):
        train_size = int(test_size_percent * len(self))
        test_size = len(self) - train_size
        x_train, x_test = torch.utils.data.random_split(self.input_images, [train_size, test_size])
        y_train, y_test = torch.utils.data.random_split(self.ground_truth, [train_size, test_size])
        return x_train, y_train, x_test, y_test


def compare_images(image_a, image_b):
    """ could be the ground truth and the outputs of a UNet or a baseline"""
    # metrics
    mean_squared = mean_squared_error(image_a, image_b)
    # ssim = structural_similarity(image, baseline_image.squeeze())  # needs to be grayscale
    psnr = peak_signal_noise_ratio(image_a, image_b)

    plot = False
    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(image_a[:, :, 0], cmap='jet')
        ax[0, 1].imshow(image_b[:, :, 0], cmap='jet')
        ax[1, 0].imshow(image_a)
        ax[1, 1].imshow(image_b)
        fig.suptitle(f"{STRIDE=} {PATCH_SIZE=}\n MSE: {mean_squared:.2f}, PSNR {psnr:.2f}dB")
        plt.show()

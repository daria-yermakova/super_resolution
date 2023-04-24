from pathlib import Path

import torch
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from image_preprocessing import get_cropped_path

stride = 4
patch_size = 512
patches_dir = Path(f"data/patches_{patch_size}/")
cropped_dir = Path(f"data/cropped/ps{patch_size}_str{stride}")  # todo inconsistent naming
if not cropped_dir.exists() and not patches_dir.exists():
    raise FileNotFoundError(
        "Make sure you've preprocessed your imagery with this configuration:"
        f"\n{stride = } \n{patch_size = }"
    )

test_patches = list(patches_dir.glob("*.jpg"))

test_patch = test_patches[2]
cropped_image_file = get_cropped_path(test_patch, cropped_dir, stride)
assert cropped_image_file.exists()

image = io.imread(test_patch)
baseline_image = io.imread(cropped_image_file)

# metrics
mean_squared = mean_squared_error(image, baseline_image)
# ssim = structural_similarity(image, baseline_image.squeeze())  # needs to be grayscale
psnr = peak_signal_noise_ratio(image, baseline_image)

plot = False
if plot:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(image[:, :, 0], cmap='jet')
    ax[0, 1].imshow(baseline_image[:, :, 0], cmap='jet')
    ax[1, 0].imshow(image)
    ax[1, 1].imshow(baseline_image)
    fig.suptitle(f"{stride=} {patch_size=}\n MSE: {mean_squared:.2f}, PSNR {psnr:.2f}dB")
    plt.show()


class UNet(torch.nn.Module):
    """Takes in patches of 128/512^2 RGB, returns 88^2"""

    def __init__(self, out_channels=2):
        super().__init__()

        # Learnable
        self.conv1A = torch.nn.Conv2d(3, 8, 3)
        self.conv1B = torch.nn.Conv2d(8, 8, 3)
        self.conv2A = torch.nn.Conv2d(8, 16, 3)
        self.conv2B = torch.nn.Conv2d(16, 16, 3)
        self.conv3A = torch.nn.Conv2d(16, 32, 3)
        self.conv3B = torch.nn.Conv2d(32, 32, 3)
        self.conv4A = torch.nn.Conv2d(32, 16, 3)
        self.conv4B = torch.nn.Conv2d(16, 16, 3)
        self.conv5A = torch.nn.Conv2d(16, 8, 3)
        self.conv5B = torch.nn.Conv2d(8, 8, 3)
        self.convfinal = torch.nn.Conv2d(8, out_channels, 1)
        self.convtrans34 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.convtrans45 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)

        # Convenience
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # Down, keeping layer outputs we'll need later.
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2)))))

        # Up, now we overwritte out in each step.
        out = torch.cat([self.convtrans34(out), l2[:, :, 4:-4, 4:-4]], dim=1)
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))
        out = torch.cat([self.convtrans45(out), l1[:, :, 16:-16, 16:-16]], dim=1)
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))

        # Finishing
        out = self.convfinal(out)

        return out


test = torch.rand(128, 128)
model = UNet().to("cpu")
model(test)
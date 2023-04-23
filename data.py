from pathlib import Path

from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from skimage import io
import matplotlib.pyplot as plt

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

# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(image[:, :, 0], cmap='jet')
ax[0, 1].imshow(baseline_image[:, :, 0], cmap='jet')
ax[1, 0].imshow(image)
ax[1, 1].imshow(baseline_image)
fig.suptitle(f"{stride=} {patch_size=}\n MSE: {mean_squared:.2f}, PSNR {psnr:.2f}dB")
plt.show()

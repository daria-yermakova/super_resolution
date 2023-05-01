from pathlib import Path

import pandas as pd
import torch
from matplotlib import rc, pyplot as plt
from tqdm import tqdm

from data import Data
from evaluate import compare_images
from srgan import UNet

# rc("font",**{"family":"sans-serif","sans-serif":["Helvetica"]})
rc("font", **{"size": 14, "family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

set_size = 20  # int(1920 / 4)
plot = True
device = "mps"

data_set = Data(data_dir="./data", n_images=set_size, load_baseline=True)

srgan_model_path = "/Users/roetzermatthias/Documents/studies/semester1/02506aia/super_resolution/runs/unet_11220105_weekender_hiccups/model.pt"
unet_model_path = "/Users/roetzermatthias/Documents/studies/semester1/02506aia/super_resolution/runs/unet_21152704_mse_output_slink/model.pt"

generator = UNet(n_filters=8).to(device)
unet = UNet(n_filters=32).to(device)

generator.load_state_dict(torch.load(srgan_model_path, map_location=torch.device(device)))
unet.load_state_dict(torch.load(unet_model_path, map_location=torch.device(device)))
means = pd.DataFrame(0, columns=["ssim", "mse", "psnr"], index=["baseline", "Unet", "SRGAN"], )
for peek_index in tqdm(range(len(data_set))):

    input_image, target_image, baseline_image = data_set[peek_index]

    srgan_logits = generator(input_image.unsqueeze(0).to(device))
    unet_logits = unet(input_image.unsqueeze(0).to(device))

    srgan_reconstruction = torch.nn.functional.tanh(
        torch.nn.functional.relu(srgan_logits)
    ).cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    unet_reconstruction = torch.nn.functional.tanh(
        torch.nn.functional.relu(unet_logits)
    ).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

    input_display = input_image.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_display = target_image.cpu().numpy().squeeze().transpose(1, 2, 0)

    assert target_display.shape == baseline_image.shape
    assert srgan_reconstruction.shape == baseline_image.shape

    gan_ssim, gan_mse, gan_psnr = compare_images(target_display, srgan_reconstruction)
    means.loc["SRGAN"] += gan_ssim, gan_mse, gan_psnr
    unet_ssim, unet_mse, unet_psnr = compare_images(target_display, unet_reconstruction)
    means.loc["Unet"] += unet_ssim, unet_mse, unet_psnr
    base_ssim, base_mse, base_psnr = compare_images(target_display, baseline_image)
    means.loc["baseline"] += base_ssim, base_mse, base_psnr
    # %%
    # 1 input 2 reconstruction 3 baseline 4 target
    if plot:
        images = [input_display, unet_reconstruction, srgan_reconstruction, baseline_image, target_display]
        titles = [
            "Low-resolution input\nStride:4",
            f"UNet Reconstruction\nSSIM: {unet_ssim:.2f} | PSNR: {unet_psnr:.2f}",
            f"SRGAN Reconstruction\nSSIM: {gan_ssim:.2f} | PSNR: {gan_psnr:.2f}",
            f"Lin.Interpolation Baseline\nSSIM: {base_ssim:.2f} | PSNR: {base_psnr:.2f}",
            "High-resolution Ground Truth\n",
        ]
        fig, ax = plt.subplots(4, len(images), figsize=(15, 8))
        for i, (image, title) in enumerate(zip(images, titles)):
            ax[0, i].imshow(image[100:200, 100:200, :])
            ax[0, i].set_title(title)
            ax[0, i].axes.get_xaxis().set_ticks([])
            ax[0, i].axes.get_yaxis().set_ticks([])

            ax[1, i].imshow(image[:, :, :])
            ax[1, i].axes.get_xaxis().set_ticks([])
            ax[1, i].axes.get_yaxis().set_ticks([])
        plt.tight_layout()
        Path("plots").mkdir(exist_ok=True)
        # plt.show()
        plt.savefig(Path("plots") / f"evaluation_{peek_index}.png", dpi=300)

means /= set_size
print(means)

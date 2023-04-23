"""
preprocessing main:
* slicing large images into smaller ones (patches)
    e.g. of size 512
* 'cropping' or up-sampling with interleave stride
    e.g. of stride 4
"""
import pathlib
from pathlib import Path

import cv2
import numpy as np
from skimage import io
from tqdm import tqdm


def patchify(input_dir, output_dir, patch_size: int = 512):
    input_images = input_dir.glob("*.jpg")
    output_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(input_images, desc="patchify"):
        image = io.imread(image_path)

        # add black padding
        updated_size = (((image.shape[0] // patch_size) + 1) * patch_size - image.shape[0]) / 2
        top, bottom, left, right = [int(updated_size)] * 4
        image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        for row in range(0, image_with_border.shape[0], patch_size):
            for column in range(0, image_with_border.shape[1], patch_size):
                out_path = output_dir / f"{image_path.stem}_patch_{row}_{column}.jpg"
                if not out_path.exists():
                    cv2.imwrite(
                        out_path.as_posix(),
                        image_with_border[row:row + patch_size, column:column + patch_size, :]
                    )


# %%
def crop_image(image: np.array, channels: int = None, stride: int = 4, linear=True):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if not channels:
        channels = image.shape[-1]
    new_size = (image.shape[1] // stride, image.shape[0] // stride, channels)
    interpolation = cv2.INTER_LINEAR if linear else cv2.BICUBIC
    # new_img = np.zeros((image.shape[0], image.shape[1], channels), dtype=image.dtype)
    new_image = np.zeros((new_size[0], new_size[1], channels), dtype=image.dtype)
    for i in range(new_size[1]):
        for j in range(new_size[0]):
            new_image[i, j] = image[i * stride, j * stride, :]
    upscale_img = cv2.resize(new_image, (image.shape[1], image.shape[0]), interpolation=interpolation)
    return upscale_img


def get_cropped_path(image_path, output_dir, stride):
    return output_dir / f"{image_path.stem}_cropped_s{stride}.jpg"


def crop_io(image_path, output_dir, stride: int, **crop_args):
    assert image_path.exists()
    _image = io.imread(image_path)
    _cropped_image = crop_image(_image, stride=stride, **crop_args)
    crop_image_path = get_cropped_path(image_path, output_dir, stride)
    if not crop_image_path.exists():
        return io.imsave(crop_image_path, _cropped_image, quality=100, check_contrast=False)


def preprocess(input_dir, output_dir, stride, patch_size):
    """todo make this main() and add arg parser:
        - input_fp
        - output_fp
        - stride
        - patch_size
    """
    input_dir = Path(input_dir)
    patches_dir = Path(output_dir).parent / f"patches_{patch_size}"
    output_dir = Path(output_dir) / f"ps{patch_size}_str{stride}"
    [path.mkdir(parents=True, exist_ok=True) for path in [patches_dir, output_dir]]

    patchify(input_dir, patches_dir, patch_size=patch_size)

    patches_files = patches_dir.glob("*.jpg")
    for image_fp in tqdm(patches_files, desc=f"cropping images with {stride=}"):
        crop_io(image_fp, output_dir=output_dir, stride=stride)


def main():
    preprocess(
        input_dir="./test",
        output_dir="./data/cropped",
        stride=4,
        patch_size=512
    )


if __name__ == '__main__':
    main()

import os
from skimage import io, color
from skimage.util import view_as_blocks
import numpy as np

path = "dataset/testing"
new_path = "cropped"

def preprocessing_crop ():
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = io.imread(os.path.join(path, filename))
            gray = color.rgb2gray(image).astype(np.uint8)
            cropped_image = view_as_blocks(gray, block_shape=(image.shape[0] // 2, image.shape[1] // 2))
            for row in range(cropped_image.shape[0]):
                for col in range(cropped_image.shape[1]):
                    new_filename = f"{filename[:-4]}_crop{row}{col}.jpg"
                    new_filepath = os.path.join(new_path, new_filename)
                    io.imsave(new_filepath, cropped_image[row][col])

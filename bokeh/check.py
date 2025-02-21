import torch.cuda as cuda
from torch import clip, float32, rand
from torchvision.utils import save_image
from torchvision import io
from model.pynet import PyNET
import json

def main():
    if cuda.is_available():
        print("INFO: CUDAは使用可能")
    else:
        print("WARNING : CUDAは使用不可")

    path = "/home/matsukawa_3/datasets/Bokeh_Simulation_Dataset/validation/"

    for i in range(5):
        image_input = io.read_image(path+f"original/{i}.jpg", io.ImageReadMode.RGB)
        image_target = io.read_image(path+f"bokeh/{i}.jpg", io.ImageReadMode.RGB)

        img_salient = (image_target.to(dtype=float32) - image_input.to(dtype=float32)) / 255.0

        img_salient = clip(img_salient, 0.0, 255.0)

        save_image(img_salient, f"bokeh/outputs/salient_t-s_{i}.png")

    i = rand((1, 4, 1024, 1024), dtype=float32)
    print("model analyzing")
    model = PyNET(level=1)
    # print(model)
    o = model(i)
    print(o.shape)

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np


# def crop_by_bounds(image: np.ndarray, bounds_coords: tuple) -> np.ndarray:
#     h_start, h_end, w_start, w_end = bounds_coords
#     return image[h_start:h_end, w_start:w_end]


# def upper_crop(img, upper_bound = 200):
#     mid = img.shape[1] // 2
#     return img[upper_bound:upper_bound + 2000, mid-1000:mid+1000]


# def crop_resize(undistorted_img: np.ndarray, bounds: tuple[int, int, int, int]):
#     """
#     Crops and resizes undistorted image to 2000x2000.

#     Parameters:
#         undistorted_img (np.ndarray): Undistorted image, original spatial size.
#     Returns:
#         np.ndarray: Cropped and resized image.
#     """
#     resized = cv2.resize(undistorted_img, dsize=(2654, 3538)) 
#     cropped = upper_crop(crop_by_bounds(resized, bounds))
#     return cropped
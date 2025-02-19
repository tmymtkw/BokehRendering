import torch.cuda as cuda
from torchvision.utils import save_image
from torchvision import io
import json

def main():
    if cuda.is_available():
        print("INFO: CUDAは使用可能")
    else:
        print("WARNING : CUDAは使用不可")

    path = "/home/matsukawa_3/datasets/nightphoto/train/raw/0"
    image = io.read_image(path+".png", io.ImageReadMode.UNCHANGED)
    image_huawei = io.read_image("/home/matsukawa_3/datasets/nightphoto/train/huawei/0.png", io.ImageReadMode.UNCHANGED)
    image_sony = io.read_image("/home/matsukawa_3/datasets/nightphoto/train/sony/0.JPG", io.ImageReadMode.UNCHANGED)
    js = None
    with open(path+".json", mode="r") as f:
        js = json.load(f)

    print(image.shape, image_huawei.shape, image_sony.shape, end="\n\n")
    print(image[0, 0:8, 0:8])
    print(image_huawei[1, 0:4, 0:4])
    for key, val in js.items():
        print(key, ":", val)

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
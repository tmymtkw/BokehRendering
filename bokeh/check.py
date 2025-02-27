import torch.cuda as cuda
from torch import clip, float32, rand, mean
from torchvision.utils import save_image
from torchvision import io
from model.pynet import PyNET
from model.blurred_borne import BlurredBorne
from model.modules import SPDC
from metrics.psnr import PSNR
from metrics.ssim import SSIM
from loss.blurred_mse import BlurredMSELoss
import json

def main():
    if cuda.is_available():
        print("INFO: CUDAは使用可能")
    else:
        print("WARNING : CUDAは使用不可")

    path = "/home/matsukawa_3/datasets/Bokeh_Simulation_Dataset/validation/"
    psnr = PSNR()
    ssim = SSIM()
    blmse = BlurredMSELoss(sigma=[2, 2.5, 3])

    for i in range(5):
        # break
        image_input = io.read_image(path+f"original/{i}.jpg", io.ImageReadMode.RGB)
        image_target = io.read_image(path+f"bokeh/{i}.jpg", io.ImageReadMode.RGB)
        image_input = image_input.to(dtype=float32) / 255.0
        image_target = image_target.to(dtype=float32) / 255.0
        image_input = image_input.unsqueeze(0)
        image_target = image_target.unsqueeze(0)
        print(image_input.shape, image_target.shape)
        save_image(image_target, f"bokeh/outputs/blur_{i}_origin.png")

        blmse(image_input, image_target, i)

        # img_salient = image_target - image_input
        # img_salient = clip(img_salient, 0.0, 255.0)
        # save_image(img_salient, f"bokeh/outputs/salient_t-s_{i}.png")

    #     input_numpy = image_input.detach().numpy().copy()
    #     target_numpy = image_target.detach().numpy().copy()
    #     p = psnr(input_numpy, target_numpy)
    #     s = ssim(image_input, image_target, device="cpu")
    #     print(p)
    #     print(mean(s))
    # print(s[0, :10, :10])

    # i = rand((16, 3, 512, 512), dtype=float32)
    # i = i.to("cuda")
    # print("input")
    # print(f"input: {cuda.max_memory_allocated() / (1024 * 1024)} MB")

    # print("model analyzing")
    # model = BlurredBorne(3)
    # model.to("cuda")
    # print(f"model: {cuda.max_memory_allocated() / (1024 * 1024)} MB")
    # print()
    # # print(model)
    # o = model(i)
    # print(f"after: {cuda.max_memory_allocated() / (1024 * 1024)} MB")

    # print(o[0].shape)

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
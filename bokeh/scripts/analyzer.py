import os
from time import sleep
from torchvision.utils import save_image
from scripts.tester import Tester

class Analyzer(Tester):
    def __init__(self):
        super().__init__()

    def Analyze(self):
        self.Debug("function called : Analyze")

        for i in range(10):
            img_input, img_target = self.dataset[i]
            img_input = img_input.to(self.device)
            img_target = img_target.to(self.device)

            sleep(1)

            # self.Display(
            #     f"{i}",
            #     f"[input] {img_input.dtype} {img_input.device} {img_input.shape}",
            #     f"[target] {img_target.dtype} {img_target.device} {img_target.shape}"
            # )

            self.DisplayStatus(0, i, 1, 10, 0.0, 0.0)

            img_input /= 255
            img_target /= 255
            img_input = img_input.unsqueeze(0).to("cpu")
            img_target = img_target.unsqueeze(0).to("cpu")
            save_image(img_input, os.path.join("./", self.cfg.GetInfo("path", "output"), f"debug_input_{i}.jpg"))
            save_image(img_target, os.path.join("./", self.cfg.GetInfo("path", "output"), f"debug_target{i}.jpg"))

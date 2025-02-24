from torch import no_grad, mean, nn
from torchvision.utils import save_image
from scripts.validator import Validator

class Tester(Validator):
    def __init__(self):
        super().__init__()

    def Test(self):
        self.Debug("-----test-----")

        accr = {"PSNR": 0, "SSIM": 0}

        with no_grad():
            for i, (img_input, img_target) in enumerate(self.dataloader[0], 1):
                img_input = img_input.to(self.cfg.GetDevice())
                img_target = img_target.to(self.cfg.GetDevice())

                ssim_before = mean(self.ssim(img_input, img_target, self.cfg.GetDevice()))
                psnr_before = self.psnr(img_input.to("cpu").detach().numpy().copy(), 
                                     img_target.to("cpu").detach().numpy().copy())

                img_output = self.model(img_input)
                # 出力の保存
                save_image(img_output, self.cfg.GetPath("output")+f"imgs/{i}.png")

                ssim_after = mean(self.ssim(img_output, img_target, self.cfg.GetDevice()))
                o = img_output.to("cpu").detach().numpy().copy()
                t = img_target.to("cpu").detach().numpy().copy()
                psnr_after = self.psnr(o, t)
                accr["SSIM"] += ssim_after
                accr["PSNR"] += psnr_after

                self.Info(f"image {i} {img_input.shape}\n"
                          + f"before SSIM : {ssim_before} PSNR : {psnr_before}\n"
                          + f"after SSIM : {ssim_after} PSNR : {psnr_after}", extra={"n": 3})
            
            self.Info(f"\033[3B[result] SSIM : {accr['SSIM']/len(self.dataset[0])} PSNR : {accr['PSNR']/len(self.dataset[0])}", extra={"n": 0})

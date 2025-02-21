from torch import save, mean, load, no_grad
# import torch
# TODO
from torch.nn import Module
from torch.utils.data import DataLoader
from util.data.bokeh_dataset import BokehDataset
from scripts.recorder import Recorder
from model.net import Net
from metrics.psnr import PSNR
from metrics.ssim import SSIM


class Trainer(Recorder):
    def __init__(self):
        super().__init__()
        
        self.mode = {"TRAIN": 0, "VALID": 1}

        self.model: Module = None

        self.train_dataset: BokehDataset = None
        self.valid_dataset: BokehDataset = None

        self.dataloader: list[DataLoader] = []

        self.optimizer = None
        self.criteria = None
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.SetDevice(self.cfg.GetInfo("option", "device"))
        self.epochs = self.cfg.GetHyperParam("epoch")

    # TODO
    def Train(self):
        """学習を行う関数

        epochs分だけProcess()を実行する

        :epochs int エポック数
        """
        self.Debug("function called: Train")

        assert self.model is not None, "\n[ERROR] model is not defined"

        for epoch in range(self.epochs):
            self.Debug("-----train--------")
            self.Process(epoch=epoch, is_train=True)

            # Validation
            if ((epoch+1) % self.cfg.GetInfo("option", "val_interval") == 0):
                self.Validate(epoch=epoch)
            
            # 重み保存
            if ((epoch+1) % self.cfg.GetInfo("option", "save_interval") or (epoch + 1) == self.epochs):
                self.PutModel(epoch)

    def Validate(self, epoch):
        self.Debug("-----validation---")
        with no_grad():
            self.Process(epoch=epoch, is_train=False)

    def Process(self, epoch, is_train=True):
        """データローダー1周分の処理を実施する関数

        is_trainがFalseの時は逆伝播を行わない

        :is_train bool 学習モードの切り替え
        """

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        loss = None
        accr = {"PSNR": 0, "SSIM": 0}
        # 学習
        for i, (img_input, img_target) in enumerate(self.dataloader[is_train], 1):
            # GPU(CPU)にデータを移動
            img_input = img_input.to(self.device, non_blocking=True)
            img_target = img_target.to(self.device, non_blocking=True)
            # self.Debug(f"{img_input.shape}")
            # self.Debug(f"{img_target.shape}")

            if is_train:
                # 勾配情報の初期化
                self.optimizer.zero_grad()

            # 順伝播
            img_output = self.model(img_input)
            # self.Debug(f"{img_output.shape}")
            # 損失の計算
            loss = self.criteria(img_output, img_target)

            if is_train:
                # 学習を行うとき
                # 逆伝播
                loss.backward()
                # オプティマイザの更新
                self.optimizer.step()
            else:
                # TODO ssim
                accr["SSIM"] = self.ssim(img_output, img_target, self.cfg.GetInfo("option", "device")).to("cpu").detach().numpy().copy()
                o = img_output.to("cpu").detach().numpy().copy()
                t = img_target.to("cpu").detach().numpy().copy()
                # self.Debug(f"input: {img_input.shape}")
                # self.Debug(f"target: {img_output.shape}")
                accr["PSNR"] += self.psnr(o, t)
            if (i == 1):
                self.Debug(f"require_grad: {img_output.requires_grad}")

            if i % self.cfg.GetInfo("option", "log_interval") == 0:
                # ターミナルに学習状況を表示
                if is_train:
                    self.DisplayStatus(epoch,
                                    i,
                                    self.epochs,
                                    len(self.train_dataset) // self.cfg.GetHyperParam("batch_size"),
                                    self.cfg.GetHyperParam("lr"),
                                    loss=loss.item())
                else:
                    self.Info(f"validating... loss : {loss.item()} PSNR : {accr['PSNR'] / i} SSIM : {accr['SSIM'].shape}", extra={ "n": 1 })
            
            
    def PutModel(self, epoch, loss=0.0):
        if self.device == "cuda":
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.to("cpu").state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f=self.cfg.GetPath("output") + f"weight_{epoch+1}.pth")
        else:
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f=self.cfg.GetPath("output") + f"weight_{epoch+1}.pth")
        
        self.Debug("model weight is saved.")

    def SetModel(self, model_class=Net):        
        self.model = model_class()

        assert isinstance(self.model, Module), f"\n[ERROR] incorrect model class: {type(model_class)}"
        self.Debug("Model created.")

        # 重みの読み込み
        # この段階ではcpu上にあるのでOK
        if self.args.weight_path is not None:
            self.Info(f"load weight: {self.args.weight_path}\n")
            weight = load(self.args.weight_path, weights_only=True)["model_state_dict"]
            for key, val in weight.items():
                self.Info(f"{key} : {val}\n")
            self.model.load_state_dict(load(self.args.weight_path, weights_only=True)["model_state_dict"])

    def SetDataset(self, img_dir, input_dir, target_dir):
        self.train_dataset = BokehDataset(self.cfg.GetPath("dataset") + self.cfg.GetPath("train"),
                                          self.cfg.GetPath("input"),
                                          self.cfg.GetPath("target"))
        self.valid_dataset = BokehDataset(self.cfg.GetPath("dataset") + self.cfg.GetPath("validation"),
                                     self.cfg.GetPath("input"),
                                     self.cfg.GetPath("target"))
        self.Debug("Dataset created.")

    def SetDataLoader(self,
                      batch_size=32,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True,
                      drop_last=True):
        # TODO: for d in self.dataset:
        train_dataloader = DataLoader(self.train_dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     drop_last=drop_last)
        valid_dataloader = DataLoader(self.valid_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=1,
                                      pin_memory=pin_memory,
                                      drop_last=False)
        self.dataloader.append(valid_dataloader)
        self.dataloader.append(train_dataloader)
        self.Debug("Dataloader created.")
        
    def SetDevice(self, device):
        assert (device == "cuda" or device == "cpu"), \
            f"\n[ERROR] incorrevt device type : {device}"
        
        self.Debug(f"setting device: {device}")
        self.device = device
    
    def DisplayStatus(self, cur_epoch, cur_itr, max_epoch, max_itr, lr=0.0, loss=0.0):
        """学習状況の標準出力

        同じフォーマットで描画を更新する
        """

        self.Info(msg="", extra={"status": {"cur_epoch": cur_epoch+1,
                                            "cur_itr": cur_itr+1,
                                            "max_epoch": max_epoch,
                                            "max_itr": max_itr,
                                            "lr": lr,
                                            "loss": loss},
                                "n": 6})
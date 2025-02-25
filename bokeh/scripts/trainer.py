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

        self.dataset: BokehDataset = []
        self.size = [0, 0]
        self.dataloader: list[DataLoader] = []

        self.optimizer = None
        self.criteria = None
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.epochs = self.cfg.GetHyperParam("epoch")

        # self.scaler = torch.cuda.amp.GradScaler("cuda")

    # TODO
    def Train(self):
        """学習を行う関数

        epochs分だけProcess()を実行する

        :epochs int エポック数
        """
        self.Debug("function called: Train")

        assert self.model is not None, "\n[ERROR] model is not defined"

        for epoch in range(1, self.epochs+1):
            self.Debug("-----train--------")
            self.Process(epoch=epoch, is_train=True)

            # Validation
            if (epoch % self.cfg.GetInfo("option", "val_interval") == 0):
                self.Validate(epoch=epoch)
            
            # 重み保存
            if (epoch % self.cfg.GetInfo("option", "save_interval") == 0 or epoch == self.epochs):
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
            # img_input = img_input.to(self.device, non_blocking=True)
            # img_target = img_target.to(self.device, non_blocking=True)
            img_input = img_input.to(self.cfg.GetDevice())
            img_target = img_target.to(self.cfg.GetDevice())
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
                accr["SSIM"] += mean(self.ssim(img_output, img_target, self.cfg.GetDevice())).item()
                o = img_output.to("cpu").detach().numpy().copy()
                t = img_target.to("cpu").detach().numpy().copy()
                # self.Debug(f"input: {img_input.shape}")
                # self.Debug(f"target: {img_output.shape}")
                accr["PSNR"] += self.psnr(o, t)
            if (i == 1):
                self.Debug(f"require_grad: {img_output.requires_grad}")

            if i == 1 or i % self.cfg.GetInfo("option", "log_interval") == 0:
                # ターミナルに学習状況を表示
                self.DisplayStatus(epoch,
                                    i,
                                    self.epochs,
                                    self.size[is_train],
                                    self.cfg.GetHyperParam("lr"),
                                    loss=loss.item())
                if not is_train:
                    self.Info(f"validating... loss : {loss.item():.12f} PSNR : {accr['PSNR']/i:.12f} SSIM : {accr['SSIM']/i:.12f}", extra={ "n": 1 })
                    
    def PutModel(self, epoch, loss=0.0):
        if self.cfg.GetDevice() == "cuda":
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.to("cpu").state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f=self.cfg.GetPath("output") + f"weight_{epoch+1}.pth")
            # .to()によるメモリ移動を戻す
            self.model.to("cuda")
        else:
            save(obj={"epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss},
                f=self.cfg.GetPath("output") + f"weight_{epoch}.pth")
        
        self.Debug(f"model at epoch {epoch} weight is saved.")

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

    def SetDataset(self
                   ):
        train_dataset = BokehDataset(self.cfg.GetPath("dataset") + self.cfg.GetPath("train"),
                                          self.cfg.GetPath("input"),
                                          self.cfg.GetPath("target"))
        valid_dataset = BokehDataset(self.cfg.GetPath("dataset") + self.cfg.GetPath("validation"),
                                     self.cfg.GetPath("input"),
                                     self.cfg.GetPath("target"),
                                     is_train=False)
        self.dataset = [valid_dataset, train_dataset]
        self.size = [len(valid_dataset), len(train_dataset) // self.cfg.GetHyperParam("batch_size")]
        self.Debug("Dataset created.")

    def SetDataLoader(self,
                      batch_size=32,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True,
                      drop_last=True):
        # TODO: for d in self.dataset:
        train_dataloader = DataLoader(self.dataset[1],
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory,
                                     drop_last=drop_last)
        valid_dataloader = DataLoader(self.dataset[0],
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=1,
                                      pin_memory=pin_memory,
                                      drop_last=False)
        self.dataloader.append(valid_dataloader)
        self.dataloader.append(train_dataloader)
        self.Debug("Dataloader created.")
    
    def DisplayStatus(self, cur_epoch, cur_itr, max_epoch, max_itr, lr=0.0, loss=0.0):
        """学習状況の標準出力

        同じフォーマットで描画を更新する
        """

        self.Info(msg="", extra={"status": {"cur_epoch": cur_epoch,
                                            "cur_itr": cur_itr,
                                            "max_epoch": max_epoch,
                                            "max_itr": max_itr,
                                            "lr": lr,
                                            "loss": loss},
                                "n": 6})
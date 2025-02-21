import os
from torch.utils.data import Dataset
from torchvision import io
from util.data.transforms import Stack, RandomFlip, RandomCrop, Convert, Scaling

class BokehDataset(Dataset):
    def __init__(self, img_dir, input_dir, target_dir, seed=42):
        super().__init__()

        assert os.path.isdir(img_dir), self.PutError(img_dir)

        self.img_dir = os.path.abspath(img_dir)
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.img_id = None
        self.SetImgId()

        self.stack = Stack(seed=seed)
        self.stack.Push(RandomCrop(height=1024, width=1024, seed=seed))
        self.stack.Push(Scaling(size=[256, 256]))
        self.stack.Push(RandomFlip(seed=seed))
        self.stack.Push(Convert(convert_type="float32"))

    def __len__(self) -> int:
        return len(self.img_id)
    
    def __getitem__(self, index):
        # 画像読み込み
        img_input = io.read_image(os.path.join(self.img_dir, self.input_dir, self.img_id[index]),
                                  io.ImageReadMode.RGB)
        img_target = io.read_image(os.path.join(self.img_dir, self.target_dir, self.img_id[index]),
                                   io.ImageReadMode.RGB)

        # データの前処理
        img_input, img_target = self.stack(img_input, img_target)

        return (img_input, img_target)
    
    def SetImgId(self):
        self.img_id = [
            file 
            for file in os.listdir(os.path.join(self.img_dir, self.input_dir))
            if self.GetIsImage(file)
        ]

        print(os.path.join(self.img_dir, self.input_dir))
        print(f"total img: {self.__len__()}\n")

    def GetIsImage(self, file):
        return (os.path.isfile(os.path.join(self.img_dir, self.input_dir, file)) 
                and os.path.isfile(os.path.join(self.img_dir, self.target_dir, file)))
    
    # TODO
    def GetSeed(self):
        return 1

    def PutError(self, path) -> str:
        return f"\n[Error] ディレクトリが見つかりません \
                 \nInput: {path} \
                 \nAbs path: {os.path.abspath(path)}\n"
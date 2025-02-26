from .net import Net
from .blurred_borne import BlurredBorne
from torch.nn import ZeroPad2d

class PadNet(Net):
    def __init__(self, model=BlurredBorne):
        super().__init__()

        self.net = model()

    def forward(self, x):
        """パディングを行う
        順伝播処理はsuper().forward(self, x)
        """
        _, _, h, w = x.shape
        t = 0, b = 0, l = 0, r = 0
        is_pad = False
        
        # pad
        if h % 8 != 0:
            is_pad = True
            pad_h = 8 - h % 8

            t = pad_h // 2
            b = 8 - t

        if w % 8 != 0:
            is_pad = True
            pad_w = 8 - w % 8

            l = pad_w // 2
            r = 8 - l

        if is_pad:
            pad = ZeroPad2d((l, r, t, b))
            x = pad(x)

        out, bokeh = self.net(x)

        if is_pad:
            out = out[:, :, t:-b, l:-r]

        return out
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        self.net.load_state_dict(state_dict, strict, assign)
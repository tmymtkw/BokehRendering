# from torch import nn
from torch import cat, clamp
from torch.nn import Module, Sequential
from torch.nn.functional import pixel_shuffle, interpolate
from .modules import FocusGenerator, SPDC3
from .modules import ConvBlock

class BlurredBorne3(Module):
    def __init__(self, img_channels=3, hidden_channels=[32, 64, 128, 256, 384]):
        super().__init__()
        # first process
        self.stem = Sequential(ConvBlock(img_channels, img_channels*4, kernel_size=7, padding=3),
                               SPDC3(img_channels*4, img_channels*8),

                               ConvBlock(img_channels*4, hidden_channels[0], kernel_size=7, padding=3),
                               SPDC3(hidden_channels[0], hidden_channels[0]*2))

        # focus attention
        # self.focus_generator = FocusGenerator()

        # down sample
        self.down_2 = ConvBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                                kernel_size=2, stride=2)
        
        self.down_4 = ConvBlock(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                                kernel_size=2, stride=2)
        
        self.down_8 = ConvBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[3],
                                kernel_size=2, stride=2)
        
        # spdc
        self.in_2 = Sequential(SPDC3(in_channels=hidden_channels[1], hidden_channels=hidden_channels[1]*2),
                               SPDC3(in_channels=hidden_channels[1], hidden_channels=hidden_channels[1]*2))

        self.in_4 = Sequential(SPDC3(in_channels=hidden_channels[2], hidden_channels=hidden_channels[2]*2),
                               SPDC3(in_channels=hidden_channels[2], hidden_channels=hidden_channels[2]*2))
        #  TODO
        self.bot = Sequential(ConvBlock(in_channels=hidden_channels[3], out_channels=hidden_channels[4], kernel_size=3, padding=1),
                              SPDC3(in_channels=hidden_channels[4], hidden_channels=hidden_channels[4]*2),
                              SPDC3(in_channels=hidden_channels[4], hidden_channels=hidden_channels[4]*2))

        self.out_4 = Sequential(SPDC3(in_channels=hidden_channels[2] + hidden_channels[4]//4,
                                     hidden_channels=hidden_channels[2]*2 + hidden_channels[4]//2, has_skip_connection=False),
                                SPDC3(in_channels=hidden_channels[2] + hidden_channels[4]//4,
                                     hidden_channels=hidden_channels[2]*2 + hidden_channels[4]//2))

        self.out_2 = Sequential(SPDC3(in_channels=hidden_channels[1] + hidden_channels[2]//4 + hidden_channels[4]//16,
                                     hidden_channels=hidden_channels[1]*2 + hidden_channels[2]//2 + hidden_channels[4]//8,
                                     has_skip_connection=False),
                                SPDC3(in_channels=hidden_channels[1] + hidden_channels[2]//4 + hidden_channels[4]//16,
                                     hidden_channels=hidden_channels[1]*2 + hidden_channels[2]//2 + hidden_channels[4]//8))

        self.bokeh_conv = ConvBlock(in_channels=hidden_channels[1] + hidden_channels[2]//4 + hidden_channels[4]//16,
                                    out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.out_conv = Sequential(SPDC3(in_channels=6, hidden_channels=6*3),
                                     ConvBlock(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # b, h[0], h, w
        feature = self.stem(x)

        # b, h[1], h/2, w/2
        bokeh_in_2 = self.down_2(feature)
        bokeh_in_2 = self.in_2(bokeh_in_2)

        # b, h[2], h/4, w/4
        bokeh_in_4 = self.down_4(bokeh_in_2)
        bokeh_in_4 = self.in_4(bokeh_in_4)

        # b, h[3], h/8, w/8
        bokeh_in_8 = self.down_8(bokeh_in_4)
        # b, h[4], h/8, w/8
        bokeh_out_8 = self.bot(bokeh_in_8)

        # b, h[2]+h[4]/4, h/4, w/4 : 64 + 96
        bokeh_out_4 = cat((bokeh_in_4, pixel_shuffle(bokeh_out_8, 2)), dim=1)
        bokeh_out_4 = self.out_4(bokeh_out_4)

        # b, h[1]+(h[2]+h[3]/4)/4, h/2, w/2 : 32+16+24
        bokeh_out_2 = cat((bokeh_in_2, pixel_shuffle(bokeh_out_4, 2)), dim=1)
        bokeh_out_2 = self.out_2(bokeh_out_2)
        # !!! this tensor must be (b, 3, h/2, w/2) : used for loss calculation
        bokeh_out_2 = self.bokeh_conv(bokeh_out_2)

        # plain upsample
        bokeh_out = interpolate(bokeh_out_2, scale_factor=2, mode="bilinear")

        # out = x * (1 - focus_weight) + bokeh_out * focus_weight
        out = cat([x, bokeh_out], dim=1)
        out = self.out_conv(out)

        return (out, bokeh_out_2)
        
    def __str__(self):
        return "BlurredBorne3"
from inspect import isfunction

from torch import nn
from einops.layers.torch import Rearrange



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2, padding=0),
    )

def Downsample(in_ch, out_ch):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(in_channels=in_ch * 4, out_channels=out_ch, kernel_size=1),
    )

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode='circular')
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class ConvBlk(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, is_last=False):
        super().__init__()
        self.block1 = Block(in_ch, mid_ch)
        self.block2 = Block(mid_ch, mid_ch)
        self.block3 = Block(mid_ch, out_ch) if not is_last\
                    else nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class DisBlk(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block1 = Block(in_ch, mid_ch)
        self.block2 = Block(mid_ch, mid_ch)
        self.block3 = Block(mid_ch, out_ch)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


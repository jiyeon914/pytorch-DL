import torch
from torch import nn
import torch.nn.functional as F

from models.module import default, ConvBlk, Upsample



class BaselineCNN(nn.Module):
    def __init__(self, channels=1, out_dim=None, dim_blocks=[64, 64, 64, 64, 32, 16], self_condition=False):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.num_blocks = len(dim_blocks)
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        
        # self.downs = nn.ModuleList([])
        # for i in range(self.num_blocks):
        #     self.downs.append(
        #         nn.ModuleList(
        #             [
        #             nn.AvgPool2d(2**i),
        #             ]
        #         )
        #     )
        
        dims = [input_channels, *map(lambda m: 1 * m, dim_blocks)]
        in_out = list(zip(dims[:-2], dims[1:-1]))

        self.ups = nn.ModuleList([])
        for idx, (in_ch, out_ch) in enumerate(in_out):
            is_first = idx == 0
            in_ch = in_ch if is_first else in_ch + 1
            self.ups.append(
                nn.ModuleList(
                    [
                    ConvBlk(in_ch, out_ch, out_ch),
                    Upsample(out_ch, out_ch),
                    ]
                )
            )
        
        self.out_dim = default(out_dim, channels)
        self.final_blk = ConvBlk(dims[-2], dims[-1], self.out_dim, is_last=True)

    def forward(self, x, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
        pooled = []
        for i in range(self.num_blocks): #in self.downs:
            pooled.append(F.avg_pool2d(x, 2**i))

        for block, up in self.ups:
            h = block(pooled.pop() if len(pooled) == self.num_blocks else torch.cat((h, pooled.pop()), dim=1))
            h = up(h)
        return self.final_blk(h)
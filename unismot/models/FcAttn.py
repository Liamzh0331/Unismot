import torch
import torch.nn as nn


class BcAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.in_channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        if x.shape[-1] != self.in_channels:
            x = x.permute(0, 2, 3, 1).contiguous()
        ori_x = x
        x = self.norm(x)
        x_global = x.mean([1, 2], keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)

        attn = c_attn
        out = ori_x * attn
        return out.permute(0, 3, 1, 2).contiguous()

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self, c1=64, c2=64):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=int(c1) // 2, oup=int(c2) // 2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=int(c1) // 2, oup=int(c2) // 2, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=int(c1) // 2, oup=int(c2) // 2, expand_ratio=2)
        self.shffleconv = nn.Conv2d(c1, c2, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DFE(nn.Module):
    # DetailFeatureExtraction
    def __init__(self, c1=64, c2=64, num_layers=3):
        super(DFE, self).__init__()
        INNmodules = [DetailNode(c1, c2) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

if __name__ == '__main__':
    inputs1 = torch.randn((2, 16, 480, 640)).cuda()
    inputs2 = torch.randn((2, 16, 480, 640)).cuda()
    inputs3 = torch.randn((2, 16, 480, 640)).cuda()
    x1 = torch.cat((inputs1, inputs2), dim=1)
    x2 = torch.cat((inputs2, inputs3), dim=1)
    x3 = torch.cat((inputs1, inputs3), dim=1)
    DEF1 = DFE(32, 32, 1).cuda()
    DEF2 = DFE(32, 32, 1).cuda()
    DEF3 = DFE(32, 32, 1).cuda()
    BcAttn = BcAttn(96).cuda()
    pred1 = DEF1(x1)
    pred2 = DEF1(x2)
    pred3 = DEF1(x3)
    x = torch.cat((pred1, pred2, pred3), dim=1)
    x = BcAttn(x)
    print(x.size())
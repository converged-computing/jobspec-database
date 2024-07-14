import math
import torch

from torch import Tensor
from torch import nn

class Discriminator(nn.Module):
    """
    A discriminator with the exact architechture described in the SRGAN paper.

    Note: doesnt have a sigmoid layer at the end - will return output from linear layer. this is by design.
    Note: only support 96x96 images - could be extended by another layer to accomdate larger images
    """

    def __init__(self, config) -> None:
        """
        Initialize discriminator

        Parameters
        ----------
        config: Config, a config object from which settings are read. in particular:
            MODEL.D_IN_CHANNEL, MODEL.D_N_CHANNEL, MODEL.D_OUT_CHANNEL
        """
        super(Discriminator, self).__init__()

        in_channels = config.MODEL.D_IN_CHANNEL
        channels = config.MODEL.D_N_CHANNEL
        out_channels = config.MODEL.D_OUT_CHANNEL

        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(channels, channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(int(2 * channels), int(2 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(int(4 * channels), int(4 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class Generator(nn.Module):
    """
    A generator module as described in SRGAN - follows the architechture outlined there.
    """

    def __init__(self, config) -> None:
        """
        Initialize generator.

        Parameters
        ----------
        config: Config, the config to read settings from. Of importance are:
            MODEL.G_IN_CHANNEL, MODEL.G_OUT_CHANNEL, MODEL.G_N_CHANNEL, MODEL.G_N_RCB, DATA.UPSCALE_FACTOR
        """
        super(Generator, self).__init__()
        in_channels: int = config.MODEL.G_IN_CHANNEL
        out_channels: int = config.MODEL.G_OUT_CHANNEL
        channels: int = config.MODEL.G_N_CHANNEL
        num_rcb: int = config.MODEL.G_N_RCB
        upscale: int = config.DATA.UPSCALE_FACTOR

        # Better weight initialization, see ESRGAN appendix and Kaming He paper "Deep Residual Learning for Image Recognition"
        # Promises better results without BN artifacts
        # TODO: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch

        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)
        return x
    

class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.rcb(x)
        x = torch.add(x, identity)
        return x



if __name__ == "__main__":
    from config import Config
    c = Config()
    d = Discriminator(c)
    g = Generator(c)
    print(f"Generator has: {sum(p.numel() for p in g.parameters())} parameters")    # 1547350 parameters
    print(f"Discriminator has: {sum(p.numel() for p in d.parameters())} parameters")# 23563649 parameters
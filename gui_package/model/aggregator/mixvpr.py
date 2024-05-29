import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn.functional import normalize
import numpy as np


@dataclass
class MixVPRConfig:
    """
    Configuration class for MixVPR model.
    * `in_channels`: int = 1024. Depth of input feature maps.
    * `in_h`: int = 20. Height of input feature maps.
    * `in_w`: int = 20. Width of input feature maps.
    * `out_channels`: int = 512. Depth wise projection dimension.
    * `mix_depth`: int = 1. The number of stacked FeatureMixers.
    * `mlp_ratio`: int = 1. Ratio of the mid projection layer in the mixer block.
    * `out_rows`: int = 4. Row wise projection dimesion.
    """

    in_channels: int = 1024
    in_h: int = 20
    in_w: int = 20
    out_channels: int = 512
    mix_depth: int = 1
    mlp_ratio: int = 1
    out_rows: int = 4


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim: int, mlp_ratio: int = 1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self, config: MixVPRConfig) -> None:
        super().__init__()
        self.__mixvpr_config = config

        hw = self.__mixvpr_config.in_h * self.__mixvpr_config.in_w

        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=self.__mixvpr_config.mlp_ratio)
                for _ in range(self.__mixvpr_config.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(
            self.__mixvpr_config.in_channels, self.__mixvpr_config.out_channels
        )
        self.row_proj = nn.Linear(hw, self.__mixvpr_config.out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = normalize(x.flatten(1), p=2, dim=-1)
        return x


# -------------------------------------------------------------------------------


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params/1e6:.3}M")


def main():
    x = torch.randn(1, 1024, 20, 20)
    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=1024,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4,
    )

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == "__main__":
    main()

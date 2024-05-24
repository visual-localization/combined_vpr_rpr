from typing import TYPE_CHECKING
from pytorch_lightning import LightningModule

from .aggregator.mixvpr import MixVPR, MixVPRConfig
from .backbone.resnet import ResNet, ResNetConfig


if TYPE_CHECKING:
    from torch import Tensor


class MixVPRModel(LightningModule):
    """This is the main model for Visual Place Recognition."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        encoder_config, agg_config, faiss_gpu = self.parse_legacy_args(*args, **kwargs)
        self.encoder_config = encoder_config
        self.agg_config = agg_config

        self.save_hyperparameters()

        self.faiss_gpu = faiss_gpu

        self.backbone = ResNet(self.encoder_config)
        self.aggregator = MixVPR(self.agg_config)

    @staticmethod
    def parse_legacy_args(
        # ---- Backbone
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=1,
        layers_to_crop=[],
        # ---- Aggregator
        agg_arch="ConvAP",  # CosPlace, NetVLAD, GeM
        agg_config={},
        # ---- Train hyperparameters
        lr=0.03,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        warmpup_steps=500,
        milestones=[5, 10, 15],
        lr_mult=0.3,
        # ----- Loss
        loss_name="MultiSimilarityLoss",
        miner_name="CustomMultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
        **kwargs,
    ) -> tuple[ResNetConfig, MixVPRConfig, bool]:
        resnet_config = ResNetConfig(
            model_name=backbone_arch,
            pretrained=pretrained,
            crop_last_layer=4 in layers_to_crop,
            crop_second_to_last_layer=3 in layers_to_crop,
            layers_to_freeze=layers_to_freeze,
        )

        mixvpr_config = MixVPRConfig(
            in_channels=agg_config["in_channels"],
            in_h=agg_config["in_h"],
            in_w=agg_config["in_w"],
            out_channels=agg_config["out_channels"],
            mix_depth=agg_config["mix_depth"],
            mlp_ratio=agg_config["mlp_ratio"],
            out_rows=agg_config["out_rows"],
        )

        return resnet_config, mixvpr_config, faiss_gpu

    def forward(self, x: "Tensor") -> "Tensor":
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self(batch)

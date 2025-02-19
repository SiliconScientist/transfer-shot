import toml
import torch
import torch.nn as nn
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from fairchem.core.models.painn import PaiNN
from fairchem.core.models.schnet import SchNetWrap
from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    EquiformerV2EnergyHead,
)
from trot.config import Config


class DNPP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        # Shape: [batch_size, output_dim]
        return self.model(data)["energy"]


class SN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        # Shape: [batch_size, output_dim]
        return self.model(data)["energy"]


class PN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        # Shape: [batch_size, output_dim]
        return self.model(data)["energy"].unsqueeze(-1)


class EQV2(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.energy_head = EquiformerV2EnergyHead(backbone)

    def forward(self, data):
        emb = self.backbone(data)
        # Shape: [batch_size, output_dim]
        return self.energy_head(data, emb)["energy"].unsqueeze(-1)


MODEL_CLASSES = {
    "dimenetpp": (DimeNetPlusPlusWrap, DNPP, "dimenetpp_all.pt"),
    "schnet": (SchNetWrap, SN, "schnet_all_large.pt"),
    "painn": (PaiNN, PN, "painn_all.pt"),
    "equiformerv2": (EquiformerV2Backbone, EQV2, "eq2_31M_ec4_allmd.pt"),
}


def load_model(name: str, config: Config) -> nn.Module:
    expert_config = toml.load("pretrained.toml")
    model_class, wrapper, weights_filename = MODEL_CLASSES[name]
    weights_path = config.paths.experts / weights_filename
    model = model_class(**expert_config[name])
    model = wrapper(model)
    weights = torch.load(
        weights_path, map_location=torch.device(config.device), weights_only=True
    )
    model.load_state_dict(weights["state_dict"], strict=False)
    model.to(config.device)
    return model


def load_experts(names: list, config: Config) -> nn.ModuleList:
    experts = nn.ModuleList([load_model(name=name, config=config) for name in names])
    return experts

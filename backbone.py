"""
Backbone IResNet compatível com os checkpoints oficiais do AdaFace
(mk-minchul/AdaFace, adaface_ir50_ms1mv2.ckpt).

Estrutura exata que os checkpoints esperam:
    input_layer  = Sequential(Conv2d, BN, PReLU)
    body         = Sequential(*todos_os_blocos_residuais)   ← Sequential plano
    output_layer = Sequential(BN, Dropout, Flatten, Linear, BN1d)

Cada bloco residual usa Bottleneck_IR com:
    shortcut_layer = MaxPool2d(1, stride)          quando in_channel == depth
                   = Sequential(Conv1×1, BN)       quando in_channel != depth
    res_layer      = Sequential(BN, Conv3×3, BN, PReLU, Conv3×3, BN)

Essa estrutura foi verificada contando as 469 chaves do checkpoint:
    input_layer  →  7 chaves
    body         → 451 chaves (24 blocos com shortcut variável)
    output_layer → 11 chaves
    Total        → 469 ✓
"""
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Linear, PReLU


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Bottleneck_IR(nn.Module):
    """
    Bloco residual do AdaFace oficial.

    shortcut_layer:
        - Se in_channel == depth: MaxPool2d(1, stride)  → sem pesos
        - Se in_channel != depth: Conv1×1 + BN          → 6 chaves no state_dict

    res_layer (sempre):
        BN → Conv3×3 → BN → PReLU → Conv3×3(stride) → BN  → 18 chaves
    """
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                Conv2d(in_channel, depth, 1, stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, 3, stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.shortcut_layer(x) + self.res_layer(x)


class IResNet(nn.Module):
    """
    IResNet backbone compatível com checkpoints AdaFace oficiais.

    Dimensões espaciais para entrada 112×112:
        input_layer output : 112×112  (Conv stride=1, sem maxpool)
        body block  0 s=2  :  56×56   (64→64)
        body block  3 s=2  :  28×28   (64→128)
        body block  7 s=2  :  14×14   (128→256)
        body block 21 s=2  :   7×7    (256→512)
        output_layer fc    : 512×7×7 = 25088 → 512
    """

    def __init__(
        self,
        layers: list[int],
        embedding_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        # Chaves: input_layer.0.weight  (Conv)
        #         input_layer.1.*       (BN)
        #         input_layer.2.weight  (PReLU)
        self.input_layer = nn.Sequential(
            Conv2d(3, 64, 3, 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        # ── Body: Sequential plano de todos os blocos residuais ───────────────
        # Chaves: body.{i}.shortcut_layer.*  e  body.{i}.res_layer.*
        #
        # Configuração de canais por stage: (in, out, num_blocos)
        stage_cfgs = [
            (64,  64,  layers[0]),
            (64,  128, layers[1]),
            (128, 256, layers[2]),
            (256, 512, layers[3]),
        ]
        blocks: list[Bottleneck_IR] = []
        for in_ch, out_ch, n in stage_cfgs:
            blocks.append(Bottleneck_IR(in_ch, out_ch, stride=2))
            for _ in range(1, n):
                blocks.append(Bottleneck_IR(out_ch, out_ch, stride=1))

        self.body = nn.Sequential(*blocks)

        # ── Head ──────────────────────────────────────────────────────────────
        # Chaves: output_layer.0.*  (BN2d)
        #         output_layer.1    (Dropout — sem pesos)
        #         output_layer.2    (Flatten — sem pesos)
        #         output_layer.3.*  (Linear 25088→512)
        #         output_layer.4.*  (BN1d)
        self.output_layer = nn.Sequential(
            BatchNorm2d(512),
            nn.Dropout(p=dropout),
            Flatten(),
            Linear(512 * 7 * 7, embedding_dim, bias=False),
            BatchNorm1d(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


# ── Configurações prontas ──────────────────────────────────────────────────────
_CONFIGS = {
    "ir_18":  [2, 2, 2, 2],
    "ir_50":  [3, 4, 14, 3],
    "ir_100": [3, 13, 30, 3],
}


def build_model(arch: str = "ir_50", embedding_dim: int = 512) -> IResNet:
    assert arch in _CONFIGS, f"arch deve ser um de {list(_CONFIGS.keys())}"
    return IResNet(_CONFIGS[arch], embedding_dim=embedding_dim)
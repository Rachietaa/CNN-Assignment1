"""Model definitions. Add new architectures to MODELS."""
import torch.nn as nn
from torchvision import models


class TNet(nn.Module):
    """Starter baseline from the assignment notebook: one conv layer + linear head."""
    def __init__(self, num_classes=16, in_channels=1, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        spatial = (img_size - 2) // 4  # 3x3 conv (no padding) then /4 pool -> 15 for 64px
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * spatial * spatial, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(in_ch, out_ch):
    """Two 3x3 conv-BN-ReLU layers (padding keeps size), then 2x2 max-pool halves it."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SceneCNN(nn.Module):
    """VGG-style CNN trained from scratch: stacked conv blocks, global average pool, small head.

    Global average pooling (instead of flattening) keeps the head tiny and makes the model
    independent of input resolution.
    """
    def __init__(self, num_classes=16, in_channels=1, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, ch = [], in_channels
        for w in widths:
            blocks.append(conv_block(ch, w))
            ch = w
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PretrainedResNet18(nn.Module):
    """ResNet-18 with ImageNet-1k weights (torchvision IMAGENET1K_V1); final layer replaced for 16 classes.

    freeze_backbone=True trains only the new final layer ("linear probe"). In that case the backbone is
    also kept in eval mode, so its BatchNorm running statistics stay the ImageNet ones instead of being
    re-estimated from our small dataset.
    """
    def __init__(self, num_classes=16, freeze_backbone=False, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = name.startswith('fc.')

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            for name, m in self.backbone.named_children():
                if name != 'fc':
                    m.eval()
        return self

    def head_parameters(self):
        return self.backbone.fc.parameters()

    def forward(self, x):
        return self.backbone(x)


class PretrainedConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny (28M params), a modern pure CNN; final layer replaced for 16 classes.

    version='v1': torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1 — supervised training on ImageNet-1k.
    version='v2': timm 'convnextv2_tiny.fcmae_ft_in1k' — self-supervised FCMAE (fully convolutional masked
                  autoencoder) pretraining on ImageNet-1k, then supervised fine-tuning on ImageNet-1k.
    Both use stochastic depth (drop_path) so that only the pretraining differs between them.
    """
    def __init__(self, num_classes=16, version='v1', drop_path=0.1):
        super().__init__()
        self.version = version
        if version == 'v1':
            self.net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                                            stochastic_depth_prob=drop_path)
            self.net.classifier[2] = nn.Linear(self.net.classifier[2].in_features, num_classes)
            self._head = self.net.classifier[2]
        elif version == 'v2':
            import timm
            self.net = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True,
                                         num_classes=num_classes, drop_path_rate=drop_path)
            self._head = self.net.get_classifier()
        else:
            raise ValueError(f'Unknown ConvNeXt version: {version}')

    def head_parameters(self):
        return self._head.parameters()

    def forward(self, x):
        return self.net(x)


MODELS = {
    'tnet': TNet,
    'scenecnn': SceneCNN,
    'resnet18': PretrainedResNet18,
    'convnext_tiny': PretrainedConvNeXtTiny,
}


def build_model(cfg, num_classes):
    model_cfg = dict(cfg['model'])
    name = model_cfg.pop('name')
    return MODELS[name](num_classes=num_classes, **model_cfg)

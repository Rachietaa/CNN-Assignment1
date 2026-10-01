"""Model definitions. Add new architectures to MODELS."""
import torch.nn as nn


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


MODELS = {
    'tnet': TNet,
}


def build_model(cfg, num_classes):
    model_cfg = dict(cfg['model'])
    name = model_cfg.pop('name')
    return MODELS[name](num_classes=num_classes, **model_cfg)

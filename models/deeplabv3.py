import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101


class DeepLab(nn.Module):

    def __init__(self):
        super(DeepLab, self).__init__()
        # define model
        self.model = deeplabv3_resnet101(num_classes = 7, progress = True)
        self.model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=8,
        kernel_size=1,
        stride=1
        )

    def forward(self, x):
        x = self.model(x)
        return x

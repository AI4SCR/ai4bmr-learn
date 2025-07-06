import torch.nn as nn
import timm

class Backbone(nn.Module):

    def __init__(self,
                 model_name: str = 'vit_small_patch16_224',
                 num_classes: int = 0,
                 global_pool: str = "token",
                 image_size: int = 224,
                 dynamic_img_size: bool = True,
                 num_channels: int = 3,
                 pretrained: bool = False
                 ):
        super().__init__()

        self.backbone = timm.create_model(model_name=model_name,
                                     num_classes=num_classes,
                                     global_pool=global_pool,
                                     img_size=image_size,
                                     dynamic_img_size=dynamic_img_size,
                                     in_chans=num_channels,
                                     pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        return x


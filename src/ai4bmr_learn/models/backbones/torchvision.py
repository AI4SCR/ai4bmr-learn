import torch.nn as nn
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor

class Backbone(nn.Module):

    def __init__(self, model_name: str = 'resnet18', weights_version: str | None = None, layer_name: str | None = None):
        super().__init__()

        self.weights_version = weights_version
        self.weights = get_model_weights(model_name)[self.weights_version] if weights_version else None

        model = get_model(model_name, weights=self.weights)

        self.layer_name: str = layer_name or self.get_default_layer_name(model_name)
        self.backbone = create_feature_extractor(model, {self.layer_name: 'features'})

    @staticmethod
    def get_default_layer_name(model_name):
        if model_name.startswith('vit_'):
            return 'encoder.ln'
        else:
            return 'avgpool'

    def forward(self, x):
        x = self.backbone(x)['features']
        return x

# backbone = Backbone()
# import torch
# backbone(torch.randn((1,3, 224, 224))).shape

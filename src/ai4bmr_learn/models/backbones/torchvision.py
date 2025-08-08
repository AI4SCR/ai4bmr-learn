import torch.nn as nn
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor


class Backbone(nn.Module):
    """Creates a feature extractor from a torchvision model.

    This class leverages torchvision's `create_feature_extractor` to produce a model
    that returns intermediate layer features instead of classification logits.
    """

    def __init__(self, model_name: str = 'resnet18', weights_version: str | None = None, layer_name: str | None = None):
        """Initializes the Backbone module.

        Args:
            model_name: The name of the torchvision model to use (e.g., 'resnet18', 'vit_b_16').
            weights_version: The specific version of the pretrained weights to load.
                If None, the model is initialized with default weights or random weights if no default is available.
            layer_name: The name of the layer from which to extract features.
                If None, a default layer is chosen based on the model architecture.
        """
        super().__init__()

        self.weights_version = weights_version
        self.weights = get_model_weights(model_name)[self.weights_version] if weights_version else None

        model = get_model(model_name, weights=self.weights)

        self.layer_name: str = layer_name or self.get_default_layer_name(model_name)
        self.backbone = create_feature_extractor(model, {self.layer_name: 'features'})
        # TODO: expose weight transforms
        # self.transform = weights.transform

    @staticmethod
    def get_default_layer_name(model_name: str) -> str:
        """Gets the default layer name for feature extraction based on the model architecture.

        For Vision Transformer (ViT) models, it uses the final layer normalization ('encoder.ln').
        For other models (typically CNNs), it uses the final average pooling layer ('avgpool').

        Args:
            model_name: The name of the model.

        Returns:
            The recommended default layer name for feature extraction.
        """
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

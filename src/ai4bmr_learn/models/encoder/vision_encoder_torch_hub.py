import lightning as L
import torch
from torchvision.models import list_models, get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor


class VisionFeatureExtractor(L.LightningModule):

    def __init__(self,
                 model_name: str,
                 feature_layer_name: str = None,
                 weights_version: str = None):
        super().__init__()
        self.save_hyperparameters()

        self.available_models = list_models()
        self.model_name = model_name
        if model_name not in self.available_models:
            raise ValueError(f'{model_name} not in available models. See: `torchvision.models.list_models()`')

        self.weights_version = weights_version or 'DEFAULT'
        self.weights = get_model_weights(model_name)[self.weights_version]

        self.model = get_model(model_name, weights=self.weights)
        self.transform = self.weights.transforms()

        # note: another way to achieve this and discard output nodes is to use
        #  from torchvision.models.feature_extraction import create_feature_extractor
        #  self.feature_extractor = create_feature_extractor(model, {'encoder.ln': 'features'}) # layer_name, out_key
        self.feature_layer_name: str = feature_layer_name or self.get_default_feature_layer_name(model_name)
        self.feature_extractor = create_feature_extractor(self.model, {self.feature_layer_name: 'features'})
        # self.hook = self.register_hook(feature_layer_name)
        # self.features: None | torch.Tensor = None

    @staticmethod
    def get_default_feature_layer_name(model_name):
        if model_name.startswith('vit_'):
            return 'encoder.ln'
        else:
            return 'avgpool'

    @staticmethod
    def get_layer_by_name(model, layer_name: str):
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in the model.")

    def register_hook(self, feature_layer_name: str):
        layer = self.get_layer_by_name(self.model, feature_layer_name)
        return layer.register_forward_hook(self.get_layer_features())

    def get_layer_features(self):
        def hook(module, input, output):
            self.features = output.detach().cpu()

        return hook

    def forward(self, x):
        features = self.feature_extractor(x)['features']
        if self.model.__class__.__name__ == 'VisionTransformer':
            # note: extract CLS token only
            features = features[:, 0]
        else:
            # note: we global pool the feature maps, one could also flatten them
            #  no influence for resnet, but for alexnet, vgg, etc. that output 2d feature maps
            assert features.ndim == 4  # B, C, H, W
            features = features.mean(dim=(2, 3))
        return features

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        features = self(x)
        batch['prediction'] = features
        return batch


VFE = VisionFeatureExtractor


class VFESeq3(VisionFeatureExtractor):

    def forward(self, imgs):
        y = []
        B, C, H, W = imgs.size()
        N_CHANNELS_TO_PAD = torch.ceil(torch.tensor(C) / 3).to(int) * 3 - C
        zeros = torch.zeros(B, N_CHANNELS_TO_PAD, H, W, device=imgs.device)
        imgs = torch.cat([imgs, zeros], dim=1)

        for i in range(0, imgs.size(1), 3):
            x = imgs[:, i:i + 3, ...]
            features = super(VFESeq3, self).forward(x)  # note: we use the forward method from the parent
            assert features.ndim == 2
            y.append(features)

        # note: we stack the embeddings for each sample along the last dim
        #   we could then use torch.flatten(a, start_dim=-2) to flatten
        return torch.stack(y, dim=-1)


class VFESeq1(VisionFeatureExtractor):

    # TODO: factor out, use_channel into __init__
    def forward(self, imgs, use_channel=0):
        y = {}
        B, C, H, W = imgs.size()
        zeros = torch.zeros(B, 2, H, W, device=imgs.device)

        for layer in range(imgs.size(1)):
            x = imgs[:, [layer], ...]
            x = torch.cat([zeros, x], dim=1)
            features = super(VFESeq3, self).forward(x)  # note: we use the forward method from the parent
            # TODO: adapt to work like VFESeq3
            y[layer] = features.squeeze()

        return y


class ResNetRepeat(VisionFeatureExtractor):

    def __init__(self,
                 num_input_channels: int,
                 model_name: str,
                 feature_layer_name: str = None,
                 weights_version: str = None):
        from torch import nn
        super().__init__(model_name=model_name, feature_layer_name=feature_layer_name, weights_version=weights_version)

        # scale factors
        factor = int(torch.tensor(num_input_channels) // 3)
        num_to_pad = num_input_channels - 3 * factor

        # input layer parameters
        name, input_layer = list(self.model.named_modules())[1]
        out_channels = input_layer.out_channels
        stride = input_layer.stride
        padding = input_layer.padding
        kernel_size = input_layer.kernel_size
        bias = input_layer.bias is not None

        # reshaped input layer weights
        weight = input_layer.weight.repeat(1, factor, 1, 1)
        weight = torch.cat([weight, input_layer.weight[:, :num_to_pad]], dim=1)

        # replace input layer
        conv1 = nn.Conv2d(num_input_channels,
                          out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # note: new normalize the summed inputs to obtain outputs in the same range as the original model
        #   if one repeats the weights and inputs, this normalization with results in the same output as non-repeated
        norm = weight.size(1) / 3
        conv1.weight = nn.Parameter(weight / norm)
        conv1.bias = input_layer.bias if bias else None
        setattr(self.model, name, conv1)

        self.feature_layer_name: str = feature_layer_name or self.get_default_feature_layer_name(model_name)
        self.feature_extractor = create_feature_extractor(self.model, {self.feature_layer_name: 'features'})

# %%
import torch
from ai4bmr_learn.models.backbones.timm import Backbone
from pathlib import Path

maev1 = '/users/amarti51/prometex/data/maev1/logs/ssl-cords2024/x3vllm0g/checkpoints/last.ckpt'
dinov1_resnet = '/users/amarti51/prometex/data/dinov1/logs/ssl-cords2024/j5wfik3n/checkpoints/last.ckpt'
dinov1_vit = '/users/amarti51/prometex/data/dinov1/logs/ssl-cords2024/zto0ui7s/checkpoints/last.ckpt'

for prefix, path in [
    ('student_backbone.backbone.', dinov1_resnet),
    ('student_backbone.backbone.', dinov1_vit)]:
    save_path = Path(path).parent / f'module={prefix[:-1]}.ckpt'

    state_dict = torch.load(path, weights_only=False)
    student_backbone = {k.replace(prefix, ''): v
                        for k,v in state_dict['state_dict'].items() if k.startswith(prefix)}
    torch.save(student_backbone, save_path)


save_path = Path(maev1).parent / f'module=backbone.backbone.ckpt'
state_dict = torch.load(maev1, weights_only=False, map_location=torch.device('cpu'))

prefix = 'backbone.tokenizer.model.'
tokenizer = {k.replace(prefix, 'patch_embed.'): v for k,v in state_dict['state_dict'].items() if k.startswith(prefix)}
prefix = 'backbone.encoder.model.'
encoder = {k.replace(prefix, ''): v for k,v in state_dict['state_dict'].items() if k.startswith(prefix)}

backbone = {**tokenizer, **encoder}

torch.save(backbone, save_path)

model = Backbone(model_name='vit_small_patch16_224', num_channels=43)
model.backbone.load_state_dict(backbone)

model = Backbone(model_name='vit_small_patch16_224', num_channels=43, ckpt_path=Path('/users/amarti51/prometex/data/maev1/logs/ssl-cords2024/x3vllm0g/checkpoints/module=backbone.backbone.ckpt'))
model = Backbone(model_name='vit_small_patch16_224', num_channels=43, ckpt_path=Path('/users/amarti51/prometex/data/dinov1/logs/ssl-cords2024/zto0ui7s/checkpoints/module=student_backbone.backbone.ckpt'))
model = Backbone(model_name='resnet18', global_pool='avg', num_channels=43, ckpt_path=Path('/users/amarti51/prometex/data/dinov1/logs/ssl-cords2024/j5wfik3n/checkpoints/module=student_backbone.backbone.ckpt'))


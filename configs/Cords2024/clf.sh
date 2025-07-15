#!/usr/bin/env bash

# resnet18, pretrained=false, freeze_backbone=true
python clis/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml

python clis/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml \
--model.backbone.init_args.pretrained=true \
--model.freeze_backbone=true

python clis/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml \
--model.backbone.init_args.pretrained=true \
--model.freeze_backbone=false

python clis/clf.py fit --config configs/Cords2024/clf-cords2024-resnet.yaml \
--model.backbone.init_args.pretrained=true \
--model.backbone.init_args.model_name=resnet50 \
--model.input_dim=2048 \
--model.freeze_backbone=false

# vit
python clis/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml \
--model.backbone.init_args.pretrained=false \
--model.freeze_backbone=true

python clis/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml \
--model.backbone.init_args.pretrained=true \
--model.freeze_backbone=true

python clis/clf.py fit --config configs/Cords2024/clf-cords2024-vit.yaml \
--model.backbone.init_args.pretrained=true \
--model.freeze_backbone=false

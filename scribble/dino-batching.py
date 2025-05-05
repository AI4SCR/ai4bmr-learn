import torch
from torch.utils.data._utils.collate import default_collate

image = torch.randn((9, 9, 3))
global_crop = torch.randn((8, 8, 3))
local_crop = torch.randn((7, 7, 3))

global_views_list = [global_crop] * 2
global_views_stack = torch.stack(global_views_list)

local_views_list = [local_crop] * 6
local_views_stack = torch.stack(local_views_list)

global_views_as_dict = { 'image': global_views_stack, 'points': [2] * 2 }
local_views_as_dict = { 'image': local_views_stack, 'points': [6] * 6 }
sample = dict(id=1, image=image, global_views=global_views_as_dict, local_views=local_views_as_dict)
batch = default_collate([sample] * 3)
batch['global_views']['image'].shape
batch['global_views']['points']

batch['local_views']['image'].shape
batch['local_views']['points']
torch.stack(batch['local_views']['points'])

global_views_as_list = [{'image': view, 'points': 2} for view in global_views_list]
local_views_as_list = [{'image': view, 'points': 6} for view in local_views_list]
sample = dict(id=1, image=image, global_views=global_views_as_list, local_views=local_views_as_list)
batch = default_collate([sample] * 3)

batch['global_views']
batch['global_views']['points']



# %%
image = batch['image']
global_views = batch['global_views']  # 2 x [{image, points}]
global_batch = default_collate(global_views)

local_views = batch['local_views']  # 6 x [{image, points}]
num_local_views = len(local_views)
local_batch = custom_collate_fn(local_views)

# sample = [dict(images = [image] * 2)] * 3
# batch = default_collate(sample)

# %%
from lightly.transforms.dino_transform import DINOTransform
import torch
from PIL.Image import Image
from torchvision import transforms
to_pil = transforms.ToPILImage()
img = to_pil(torch.randn((3, 256, 256)))
transform = DINOTransform()
sample = transform(img)

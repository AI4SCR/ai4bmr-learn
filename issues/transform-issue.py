from torchvision.transforms import v2
import torch
import numpy as np
from torchvision import tv_tensors
from PIL import Image

numpy_image = np.random.randn(224, 224, 3)
torch_image = torch.from_numpy(numpy_image)
tv_image = tv_tensors.Image(torch_image)

uint_image = np.random.randint(0, 255, (224, 224, 3)).astype('uint8')
pil_image = Image.fromarray(uint_image)

transform = v2.Compose([
    # v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

transform(numpy_image).max()
transform(numpy_image).shape

transform(torch_image.to(torch.uint8)).max()
transform(torch_image.to(torch.uint8)).min()
transform(torch_image).shape

tv_image.max()
transform(tv_image).max()
transform(tv_image).shape
transform(tv_image).dtype

transform(uint_image).max()
transform(uint_image).shape
transform(uint_image).dtype

transform(pil_image).max()

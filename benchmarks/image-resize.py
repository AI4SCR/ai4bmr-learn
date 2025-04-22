# %%
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as v2

# Create dummy image (e.g., 512x512 RGB)
dummy_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
patch = Image.fromarray(dummy_array)

# Target size
patch_width, patch_height = 256, 256


num_iters = 100

# --- PIL Resize ---
start_pil = time.time()
for _ in range(num_iters):
    resized_pil = patch.resize((patch_width, patch_height))
    arr1 = torch.tensor(np.asarray(patch)).permute(2, 0, 1) / 255
end_pil = time.time()
print(f"PIL resize time (100 runs): {end_pil - start_pil:.4f} seconds")

# --- torchvision v2 Resize ---
transform_v2 = v2.Compose([
    v2.ToImage(),  # assumes input is PIL or ndarray
    v2.Resize((patch_height, patch_width)),
    v2.ToDtype(torch.float32, scale=True),
])

start_v2 = time.time()
for _ in range(num_iters):
    arr2 = transform_v2(patch)
end_v2 = time.time()
print(f"torchvision v2 resize time (100 runs): {end_v2 - start_v2:.4f} seconds")


transform_v3 = v2.Resize((patch_height, patch_width))

start_v3 = time.time()
for _ in range(num_iters):
    arr3 = torch.tensor(np.asarray(patch)).permute(2, 0, 1)
    arr3 = transform_v3(arr3)
    arr3 = arr3 / 255
end_v3 = time.time()
print(f"torchvision v3 array resize time (100 runs): {end_v3 - start_v3:.4f} seconds")

# --- torchvision v4 Resize ---
transform_v4 = v2.Compose([
    v2.Resize((patch_height, patch_width)),
    v2.ToImage(),  # assumes input is PIL or ndarray
    v2.ToDtype(torch.float32, scale=True),
])

start_v4 = time.time()
for _ in range(num_iters):
    arr4 = transform_v4(patch)
end_v4 = time.time()
print(f"torchvision v4 resize time (100 runs): {end_v4 - start_v4:.4f} seconds")

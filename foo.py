import os
import shutil
import random
import numpy as np
from transformers import CLIPVisionModel, AutoProcessor

backbone = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32", output_hidden_states=True
)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
import torch

a = np.empty((3, 518, 518))
b = processor(images=a, return_tensor="pt")
print(b["pixel_values"][0].shape)
c = torch.tensor(b["pixel_values"][0]).unsqueeze(0)
d = backbone(c)
print(d[0].shape)

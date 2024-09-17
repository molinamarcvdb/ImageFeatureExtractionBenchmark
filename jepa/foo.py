from src.models.vision_transformer import VisionTransformer
import torch

model = VisionTransformer.from_pretrained("nielsr/vit-large-patch16-v-jepa")
print(dir(model))
image = torch.empty([1, 1,  224, 224, 1])

feature = model(image)

print(feature.shape)

from PIL import Image
from paq2piq.paq2piq.inference_model import *

image = Image.open("/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/synthetic/diffusion/000000_calc.png")
model = InferenceModel(RoIPoolModel(), load_model_from_hub('molinamarc/syntheva', 'RoIPoolModel-fit.10.bs.120.pth'))
output = model.predict_from_pil_image(image)
print(output)
import os
import shutil
import random

case_dir = (
    "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CheXpert-v1.0-small/ImageSubfolder"
)

list_files = []

for root, dire, files in os.walk(case_dir):
    if len(files) > 0:
        list_files.extend(
            [
                os.path.join(root, file)
                for file in files
                if file.endswith((".jpg", ".jpeg", ".png"))
            ]
        )


selected_files = random.sample(list_files, 20000)

dest_dir = "data/real_chestxpert"
os.makedirs(dest_dir, exist_ok=True)
#
for file in selected_files:
    shutil.copyfile(file, os.path.join(dest_dir, os.path.basename(file)))

d = {"Filename": [], "id": []}

for i, file in enumerate(sorted(os.listdir(dest_dir))):
    d["Filename"].append(file)
    d["id"].append(i)

import pandas as pd

pd.DataFrame(d).to_csv("data/chestxpert_small.csv")

# diff_dir = '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CheXpert-v1.0-small/syntheva/logs/000-DiT-XL-2'
# dest_img_dir = 'data/diffusion_chestxpert'
# os.makedirs(dest_img_dir, exist_ok=True)
# for i, obj in enumerate(os.listdir(diff_dir)):
#    if os.path.isdir(os.path.join(diff_dir, obj)) and obj.startswith('DiT'):
#        for file in os.listdir(os.path.join(diff_dir, obj)):
#            shutil.copyfile(os.path.join(diff_dir, obj, file), os.path.join(dest_img_dir, str(i) + '_' + file))

diff_dir = "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/diffusion_chestxpert"

for file in os.listdir(diff_dir):
    os.rename(
        os.path.join(diff_dir, file),
        os.path.join(diff_dir, os.path.splitext(file)[0] + ".jpg"),
    )


# json_dir = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench/checkpoints/20240929_023653_ntxent_dino/20240929_023653_ntxent_dino_image_paths.json'
#
# import pandas as pd
#
#
# df = pd.read_csv('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/Notebooks/dicom_metadata.csv')
#
# df['Study Date'] = [int(date[:4]) for date in df['Study Date'].astype(str).tolist()]
#
#
# df = df[df['Study Date']>2021]
#
# import json
# with open(json_dir, 'r') as f:
#    js_data = json.load(f)
#
# train_paths = js_data['train_paths']
# i=0
# for path in train_paths:
#
#    dic_path = os.path.splitext(os.path.basename(path))[0] + '.dcm'
#
#    if dic_path in df['Filename'].tolist():
#        print(dic_path)
#        i +=1
# print(i)

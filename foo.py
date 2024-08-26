import os
import numpy as np
# Link azure to local path and group answers
def link_azure_local(jsonl_path):
    import json
    import os

    # Function to read JSONL file
    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield json.loads(line)

    # Load JSONL data
    data = list(load_jsonl(jsonl_path))

    # Define path mappings
    dict_map = {
        'synth_diff_calc':'./data/synthetic/diffusion',
        'synth_diff_normal':'./data/synthetic/diffusion',
        'synth_gan_calc':'./data/synthetic/gan',
        'synth_gan_normal':'./data/synthetic/gan',
        'real_normal':'./data/real',
        'real_calc':'./data/real'
    }

    # Group assessments by filename
    grouped_data = {}

    for i, record in enumerate(data):
        azure_path = record['image_path']
        category = record['category']

        # Extract filename from the Azure path
        filename = azure_path.split('/')[6].split('?')[0]

        # Generate the local file path based on the category
        if category in dict_map:
            local_path = None
            if category == 'synth_gan_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.jpeg')
            elif category == 'synth_diff_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.png')
            elif category == 'synth_gan_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.jpeg')
            elif category == 'synth_diff_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.png')
            else:
                local_path = os.path.join(dict_map[category], filename)

            if local_path:
                # Check if the local file exists
                if os.path.exists(local_path):
                    record['local_path'] = local_path
                    grouped_data[i] = record
                else: print(local_path)
    return grouped_data

def get_sets_content():
    case_dir = os.path.join('./data/features', timestamp)

    net_sets_dict = {}
    #Itreate over networks
    for network in os.listdir(case_dir):
        # Iterate over sets
        for i, file in enumerate([file for file in os.listdir(os.path.join(case_dir, network, 'synthetic')) if file.endswith('filenames.npy')]):
            actual_set = i+1
            list_files_set = np.load(os.path.join(case_dir, network, 'synthetic', file))
            net_sets_dict[actual_set] = list(list_files_set)
        break
    
    return net_sets_dict


def get_realism_set_dict(grouped_data, net_sets_dict):

    dict_sets_realism = {i+1: [] for i in range(len(net_sets_dict.keys()))}

    for i in grouped_data:
        local_path = grouped_data[i]['local_path']
        for j in net_sets_dict:
            if local_path in net_sets_dict[j]:
                dict_sets_realism[j].append(grouped_data[i]['realism_score'])
                break
    return dict_sets_realism

jsonl_path = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/turing_tests/evaluations_4e26daae-530d-430b-81be-107704de6a9e_MARC_M.jsonl'
project_dir = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench'
timestamp = '20240821_123534'

# Azure to local image path linking
grouped_data = link_azure_local(jsonl_path)

# Record of each image linking it with the local_path
# All images sets should contain the same images only iterating for first network
net_sets_dict = get_sets_content()

# Get realism dictioanry as qualitative metric in dict format
dict_sets_realism = get_realism_set_dict(grouped_data, net_sets_dict)
print(dict_sets_realism)

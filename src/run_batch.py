import os
import json
from tqdm import tqdm

dataset = 'None'
adata_left_path = 'None'
adata_right_path = 'None'

cases = []

# datasets = ['King_fixed']

# sample_pairs = {
#     'King_fixed': [['Sham_1', '1hr'], ['Sham_1', '4hr'], ['Sham_1', 'D3_1'], ['Sham_1', 'D3_3'], ['Sham_1', 'D7_2'], ['Sham_1', 'D7_3']]
# }

# sample_alpha_map = {
#     '1hr': 0.001,
#     '4hr': 0.001,
#     'D3_1': 0.001,
#     'D3_3': 0.001,
#     'D7_2': 0.001,
#     'D7_3': 0.001,
# }
# sample_lambda_map = {
#     '1hr': 0.1,
#     '4hr': 0.1,
#     'D3_1': 0.1,
#     'D3_3': 0.1,
#     'D7_2': 0.1,
#     'D7_3': 0.1,
# }

# datasets = ['Michael_T_Eadon']

# sample_pairs = {
#     'Michael_T_Eadon': [['Sham', 'CLP']]
# }

# sample_alpha_map = {
#     'Sham': 0.01,
#     'IRI': 0.001,
#     'CLP': 0.01
# }
# sample_lambda_map = {
#     'Sham': 0.075,
#     'IRI': 0.01,
#     'CLP': 0.075
# }

# datasets = ['ST_SN_AAV_MiceLiver']
# sample_pairs = {
#     'ST_SN_AAV_MiceLiver': [['Control', 'Diseased']]
# }

# sample_alpha_map = {
#     'Control': 0.0001,
#     'Diseased': 0.0001
# }
# sample_lambda_map = {
#     'Control': 0.05,
#     'Diseased': 0.05
# }

# datasets = ['Jason_Guo']

# sample_pairs = {
#     'Jason_Guo': [['Control', 'D14_1'], ['Control', 'D14_2'], ['Control', 'D14_3']]
# }

# sample_alpha_map = {
#     'Control': 0.0001,
#     'D14_1': 0.0001,
#     'D14_2': 0.0001,
#     'D14_3': 0.0001
# }
# sample_lambda_map = {
#     'Control': 0.002,
#     'D14_1': 0.002,
#     'D14_2': 0.005,
#     'D14_3': 0.005
# }


# datasets = ['Feldmans_Lab']
# sample_pairs = {
#     'Feldmans_Lab': [['WT_1_1', '5XFAD_1_1'], ['WT_1_1', 'hNSC_1_1'], ['WT_1_1', 'Vehicle_1_1']]
# }

# sample_alpha_map = {
#     'WT_1_1': 0.0001,
#     '5XFAD_1_1': 0.0001,
#     'hNSC_1_1': 0.0001,
#     'Vehicle_1_1': 0.0001
# }
# sample_lambda_map = {
#     'WT_1_1': 0.01,
#     '5XFAD_1_1': 0.01,
#     'hNSC_1_1': 0.01,
#     'Vehicle_1_1': 0.01
# }


# datasets = ['Man_Luo']
# sample_pairs = {
#     'Man_Luo': [['C1', 'P1']]
# }
# sample_alpha_map = {
#     'C1': 0.0001,
#     'P1': 0.0001
# }
# sample_lambda_map = {
#     'C1': 0.01,
#     'P1': 0.01
# }

# datasets = ['John_Roger']
# sample_pairs = {
#     'John_Roger': [['healthy', 'pod_36']]
# }
# sample_alpha_map = {
#     'healthy': 0.0001,
#     'pod_36': 0.0001
# }
# sample_lambda_map = {
#     'healthy': 0.05,
#     'pod_36': 0.05
# }

datasets = ['Duchenne_mouse_models']
sample_pairs = {
    'Duchenne_mouse_models': [['C57BL10', 'MDX']]
}
sample_alpha_map = {
    'C57BL10': 0.0001,
    'MDX': 0.0001
}
sample_lambda_map = {
    'C57BL10': 0.001,
    'MDX': 0.001
}

dissimilarities = ['js']
sinkhorn_options = [1]

for dataset in datasets:
    for sample_pair in sample_pairs[dataset]:
        sample_left = sample_pair[0]
        sample_right = sample_pair[1]
        for dissimilarity in dissimilarities:
            alpha_options = [sample_alpha_map[sample_right]]
            lambda_options = [sample_lambda_map[sample_right]]
            for sinkhorn in sinkhorn_options:
                if sinkhorn == 1:
                    for alpha in alpha_options:
                        for lambda_sinkhorn in lambda_options:
                            cases.append({
                                'dataset': dataset,
                                'sample_left': sample_left,
                                'sample_right': sample_right,
                                'dissimilarity': dissimilarity,
                                'sinkhorn': sinkhorn,
                                'alpha': alpha,
                                'lambda_sinkhorn': lambda_sinkhorn
                            })
                else:
                    for alpha in alpha_options:
                        cases.append({
                            'dataset': dataset,
                            'sample_left': sample_left,
                            'sample_right': sample_right,
                            'dissimilarity': dissimilarity,
                            'sinkhorn': sinkhorn,
                            'alpha': alpha,
                            'lambda_sinkhorn': 1
                        })

for case in tqdm(cases):
    mode = 1
    dataset = case['dataset']
    sample_left = case['sample_left']
    sample_right = case['sample_right']
    lambda_sinkhorn = case['lambda_sinkhorn']
    sinkhorn = case['sinkhorn']
    dissimilarity = case['dissimilarity']
    alpha = case['alpha']
    numIterMaxEmd = 1000000
    numInnerIterMax = 10000
    init_map_scheme = "uniform"
    use_gpu = 1
    QC = 0

    config = {
        "mode": mode,
        "dataset": dataset,
        "sample_left": sample_left,
        "sample_right": sample_right,
        "adata_left_path": adata_left_path,
        "adata_right_path": adata_right_path,
        "adata_healthy_right_path": 'None',
        "sinkhorn": sinkhorn,
        "lambda_sinkhorn": lambda_sinkhorn,
        "dissimilarity": dissimilarity,
        "alpha": alpha,
        "init_map_scheme": init_map_scheme,
        "numIterMaxEmd": numIterMaxEmd,
        "numInnerIterMax": numInnerIterMax,
        "use_gpu": use_gpu,
        "QC": QC,
        "data_folder_path": "../Data",
        "results_path": "../results",
        "grid_search": 0,
    }

    config_file_name = f'config_{dataset}_{sample_left}_vs_{sample_right}_{dissimilarity}'
    if sinkhorn:
        config_file_name += f'_sinkhorn_lambda_{lambda_sinkhorn}_alpha_{alpha}.json'
    else:
        config_file_name += f'_alpha_{alpha}.json'

    config_path = f'../configs/{config_file_name}'

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    os.system(f'python main.py --config {config_path}')

    with open(config_path) as f:
        config = json.load(f)

    config['mode'] = 2

    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)

    os.system(f'python main.py --config {config_path}')


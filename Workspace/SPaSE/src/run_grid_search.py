import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='SPaSE')
parser.add_argument('-d', '--dataset')
parser.add_argument('-l', '--adata_left_path')
parser.add_argument('-hr', '--adata_healthy_right_path')
parser.add_argument('-r', '--adata_right_path')
parser.add_argument('-g', '--use_gpu', default=0)
parser.add_argument('-ds', '--dissimilarity', default='js')

args = parser.parse_args()

alphas = [0.0001, 0.001, 0.01, 0.1]
lambda_sinkhorns = [0.001, 0.01, 0.1]

sample_left = args.adata_left_path.split('/')[-1].split('.')[0]
sample_right = args.adata_right_path.split('/')[-1].split('.')[0]

for alpha in alphas:
    for lambda_sinkhorn in lambda_sinkhorns:
        config = {
            "mode": 1,
            "dataset": args.dataset,
            "sample_left": sample_left,
            "sample_right": sample_right,
            "adata_left_path": args.adata_left_path,
            "adata_right_path": args.adata_right_path,
            "adata_healthy_right_path": args.adata_healthy_right_path,
            "sinkhorn": 1,
            "lambda_sinkhorn": lambda_sinkhorn,
            "dissimilarity": args.dissimilarity,
            "alpha": alpha,
            "init_map_scheme": 'uniform',
            "numIterMaxEmd": 1000000,
            "numInnerIterMax": 10000,
            "use_gpu": int(args.use_gpu),
            "QC": 0,
            "data_folder_path": "../Data",
            "results_path": "../results",
            "grid_search": 1,
        }

        config_file_name = f'config_{args.dataset}_{sample_left}_vs_{sample_right}_{args.dissimilarity}'
        if sinkhorn: config_file_name += f'_sinkhorn_lambda_{lambda_sinkhorn}_alpha_{alpha}.json'
        else:
            config_file_name += f'_alpha_{alpha}.json'

        config_path = f'../configs/{config_file_name}'

        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(f'{config_path}', 'w') as config_file:
            json.dump(config, config_file, indent=4)

        os.system(f'python ../main.py --config ../configs/{config_file_name}')

        with open(f'../configs/{config_file_name}') as f:
            config = json.load(f)

        config['mode'] = 2

        with open(f'{config_path}', 'w') as config_file:
            json.dump(config, config_file, indent=4)

        os.system(f'python ../main.py --config ../configs/{config_file_name}')

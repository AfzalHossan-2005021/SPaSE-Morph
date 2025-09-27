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
parser.add_argument('-sh', '--sinkhorn', default=1)
parser.add_argument('-ims', '--init_map_scheme', default='uniform')
parser.add_argument('-nim', '--numInnerIterMax', default=10000)
parser.add_argument('-nem', '--numIterMaxEmd', default=1000000)
parser.add_argument('-ifp', '--data_folder_path', default='../Data')
parser.add_argument('-rp', '--results_path', default='../results')
parser.add_argument('-qc', '--QC', default=0)


args = parser.parse_args()

alphas = [0.0001, 0.001, 0.01, 0.1]
lambda_sinkhorns = [0.001, 0.01, 0.1]

sample_left = args.adata_left_path.split('/')[-1].split('.')[0]
sample_right = args.adata_right_path.split('/')[-1].split('.')[0]

for alpha in alphas:
    for lambda_sinkhorn in lambda_sinkhorns:
        config = {
            "grid_search": 1,
            "alpha": alpha,
            "lambda_sinkhorn": lambda_sinkhorn,
            "mode": 1,
            "dataset": args.dataset,
            "sample_left": sample_left,
            "sample_right": sample_right,
            "adata_left_path": args.adata_left_path,
            "adata_right_path": args.adata_right_path,
            "adata_healthy_right_path": args.adata_healthy_right_path,
            "sinkhorn": int(args.sinkhorn),
            "dissimilarity": args.dissimilarity,
            "init_map_scheme": args.init_map_scheme,
            "numIterMaxEmd": int(args.numIterMaxEmd),
            "numInnerIterMax": int(args.numInnerIterMax),
            "use_gpu": int(args.use_gpu),
            "QC": int(args.QC),
            "data_folder_path": args.data_folder_path,
            "results_path": args.results_path,
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

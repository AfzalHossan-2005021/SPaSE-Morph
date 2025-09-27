import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='SPaSE')
parser.add_argument('-df', '--data_folder_path', default='../Data')
parser.add_argument('-ds', '--dataset')
parser.add_argument('-l', '--healthy')
parser.add_argument('-r', '--diseased')
parser.add_argument('-hr', '--healthy_right_path', default='None')
parser.add_argument('-a', '--alpha')
parser.add_argument('-ls', '--lambda_sinkhorn')
parser.add_argument('-g', '--use_gpu', default=1)


args = parser.parse_args()

healthy_path = args.healthy
diseased_path = args.diseased
adata_healthy_right_path = args.healthy_right_path

alpha = float(args.alpha)
lambda_sinkhorn = float(args.lambda_sinkhorn)
use_gpu = int(args.use_gpu)

adata_left_path = f'{args.data_folder_path}/{args.dataset}/{healthy_path}'
adata_right_path = f'{args.data_folder_path}/{args.dataset}/{diseased_path}'

sample_left = adata_left_path.split('/')[-1].split('.')[0]
sample_right = adata_right_path.split('/')[-1].split('.')[0]

sinkhorn = 1
dissimilarity = 'js'

mode = 1

numIterMaxEmd = 1000000
numInnerIterMax = 10000
init_map_scheme = "uniform"
QC = 0

config = {
    "mode": mode,
    "data_folder_path": args.data_folder_path,
    "dataset": args.dataset,
    "sample_left": sample_left,
    "sample_right": sample_right,
    "adata_left_path": adata_left_path,
    "adata_right_path": adata_right_path,
    "adata_healthy_right_path": adata_healthy_right_path,
    "sinkhorn": sinkhorn,
    "lambda_sinkhorn": lambda_sinkhorn,
    "dissimilarity": dissimilarity,
    "alpha": alpha,
    "init_map_scheme": init_map_scheme,
    "numIterMaxEmd": numIterMaxEmd,
    "numInnerIterMax": numInnerIterMax,
    "use_gpu": use_gpu,
    "QC": QC,
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

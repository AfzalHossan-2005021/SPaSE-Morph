import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='SPaSE')
parser.add_argument('-d', '--dir_name')
parser.add_argument('-l', '--healthy')
parser.add_argument('-r', '--diseased')
parser.add_argument('-hr', '--healthy_right_path', default='None')
parser.add_argument('-a', '--alpha')
parser.add_argument('-ls', '--lambda_sinkhorn')
parser.add_argument('-g', '--use_gpu', default=1)

args = parser.parse_args()


dataset = args.dir_name
healthy_path = args.healthy
diseased_path = args.diseased
adata_healthy_right_path = args.healthy_right_path

alpha = float(args.alpha)
lambda_sinkhorn = float(args.lambda_sinkhorn)
use_gpu = int(args.use_gpu)

adata_left_path = f'../../../Data/{dataset}/{healthy_path}'
adata_right_path = f'../../../Data/{dataset}/{diseased_path}'

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
    "dataset": dataset,
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
    "data_folder_path": "../../../Data",
    "sample_left_hvg_h5_save_path": f"../../../Data/{dataset}/Preprocessed",
    "sample_right_hvg_h5_save_path": f"../../../Data/{dataset}/Preprocessed",
    "results_path": "../../../Workspace/SPaSE/results",
    "grid_search": 0,
}

config_file_name = f'config_{dataset}_{sample_left}_vs_{sample_right}_{dissimilarity}'
if sinkhorn: config_file_name += f'_sinkhorn_lambda_{lambda_sinkhorn}_alpha_{alpha}.json'
else:
    config_file_name += f'_alpha_{alpha}.json'

config_path = f'../../../Workspace/SPaSE/configs/{config_file_name}'

os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(f'{config_path}', 'w') as config_file:
    json.dump(config, config_file, indent=4)

os.system(f'python ../../../Workspace/SPaSE/main.py --config ../../../Workspace/SPaSE/configs/{config_file_name}')

with open(f'../../../Workspace/SPaSE/configs/{config_file_name}') as f:
    config = json.load(f)

config['mode'] = 2

with open(f'{config_path}', 'w') as config_file:
    json.dump(config, config_file, indent=4)

os.system(f'python ../../../Workspace/SPaSE/main.py --config ../../../Workspace/SPaSE/configs/{config_file_name}')

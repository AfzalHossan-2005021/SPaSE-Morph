
import scanpy as sc
import numpy as np
from scipy import sparse
import random
import math
import torch

import argparse

parser = argparse.ArgumentParser(prog='SPaSE_perturb')
# parser.add_argument('-d', '--dataset')
# parser.add_argument('-l', '--adata_left_path')
parser.add_argument('-a', '--adata_path')
parser.add_argument('-s', '--adata_save_path')

args = parser.parse_args()

# data_folder_path = '../../../../Data'

# dataset = 'King'
# root_data_folder_name = 'Fixed_adata'
# sample_name = 'Sham_1'

adata = sc.read(args.adata_path)
adata.var_names_make_unique()

def get_to_be_remodeled_regions(adata, rad, center_idx = None):
    '''
    adata -> sample where perturbation will be added
    rad   -> radius of perturbed region
    '''

    coor = adata.obsm['spatial']
    if center_idx == None:
        center_idx = random.randint(0,adata.n_obs - 1)
    center_coor = coor[center_idx]
    print('center_idx:', center_idx)
    remodeled_idxs = [idx for idx in range(adata.n_obs) if math.dist(coor[idx], coor[center_idx]) < rad]
    
    return remodeled_idxs


def perturb(count_mat, matrix, std_multiplier=2):
    col_sums_cnt = np.sum(count_mat, axis=0)
    non_zero_cols_idxs_cnt = np.where(col_sums_cnt != 0)[0]

    count_mat_refined = count_mat[:, non_zero_cols_idxs_cnt]
    
    means = torch.from_numpy(count_mat_refined.mean(axis=0)).cuda()
    stds = torch.from_numpy(count_mat_refined.std(axis=0)).cuda()
    samples = torch.randn((matrix.shape[0], len(means))).cuda() * stds * std_multiplier + means
    if torch.any(torch.isnan(samples)):
        print("nan found :(")
    matrix_with_noise = matrix.copy()
    matrix_with_noise[:, non_zero_cols_idxs_cnt] = samples.cpu().numpy()
    matrix_with_noise[matrix_with_noise < 0] = 0
    return matrix_with_noise


scale = 2


x = adata.obsm['spatial'][:, 0]
y = adata.obsm['spatial'][:, 1]

center_idx = random.randint(0, adata.n_obs - 1)
adata.obs['is_remodeled_for_grid_search'] = 0
try:
    count_mat_perturbed = adata.X.toarray()
except:
    count_mat_perturbed = adata.X

# for d in [2, 3, 4]:
rad = min(y.max() - y.min(), x.max() - x.min()) / 4
remodeled_idxs = get_to_be_remodeled_regions(adata, rad, center_idx=center_idx)
remodeled_barcodes = adata.obs.index[remodeled_idxs]
adata.obs.loc[remodeled_barcodes, 'is_remodeled_for_grid_search'] = 1
try:
    count_mat_perturbed[remodeled_idxs] = perturb(adata.X.toarray(), adata.X.toarray()[remodeled_idxs], std_multiplier=scale)
except:
    count_mat_perturbed[remodeled_idxs] = perturb(adata.X, adata.X[remodeled_idxs], std_multiplier=scale)


count_mat_perturbed_sparse = sparse.csr_matrix(count_mat_perturbed)
adata.X = count_mat_perturbed_sparse

adata.write(args.adata_save_path)
# adata.write(f'{data_folder_path}/{dataset}/Fixed_adatas/adata_{sample_name}_type_{type}_orig_perturb_centering_{center_idx}.h5ad')

# remodeled_barcodes_orig = remodeled_barcodes

# plt.figure(figsize=(8, 8))
# color_map = {
#     0: 'blue',
#     1: 'red',
#     2: 'green',
#     3: 'violet',
#     4: 'black',
# }
# spot_label = adata.obs['is_remodeled_for_grid_search']
# plt.scatter(adata.obsm['spatial'][:, 0], -adata.obsm['spatial'][:, 1], c=list(map(lambda x:color_map[x], spot_label)))
# plt.savefig(f'/home/nuwaisir/Corridor/Samee_sir_lab/Data/King/Perturbed_adata_figures/sample_{sample_name}_type_{type}_sample_num_{sample_num}.jpg')




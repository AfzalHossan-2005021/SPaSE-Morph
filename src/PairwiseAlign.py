import os
import ot
import torch
import numpy as np
import scanpy as sc
from DataLoader import DataLoader
from utils import scale_coords, QC, paste_pairwise_align_modified


class PairwiseAlign():
    def __init__(self, config):
        self.config = config
        self.dataset = config['dataset']
        self.sample_left = config['sample_left']
        self.sample_right = config['sample_right']
        self.alpha = config['alpha']
        self.dissimilarity = config['dissimilarity']
        self.init_map_scheme = config['init_map_scheme']
        self.numIterMaxEmd = config['numIterMaxEmd']
        self.numInnerIterMax = config['numInnerIterMax']
        self.use_gpu = config['use_gpu']
        self.results_path = config['results_path']
        self.config_file_name = os.path.basename(config['config_path'])
        self.sinkhorn = config['sinkhorn']
        self.lambda_sinkhorn = config['lambda_sinkhorn']
        self.cost_mat_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/cost_mat_{self.sample_left}_{self.sample_right}_{self.dissimilarity}.npy'
        os.makedirs(os.path.dirname(self.cost_mat_path), exist_ok=True)

        if config['adata_left_path'] != 'None':
            self.adata_left = sc.read(config['adata_left_path'])
            self.adata_right = sc.read(config['adata_right_path'])
        else:
            data_loader = DataLoader(config)

            dataset_map = data_loader.read_data(self.dataset)

            self.adata_left = dataset_map[self.sample_left]
            self.adata_right = dataset_map[self.sample_right]

        scale_coords(self.adata_left, key_name='spatial')
        scale_coords(self.adata_right, key_name='spatial')

        if config['QC']:
            QC(self.adata_left)
            QC(self.adata_right)

    def pairwise_align_sinkhorn(self):
        if not torch.cuda.is_available():
            if self.use_gpu:
                print("Setting use_gpu to False")
            self.use_gpu = False

        if self.use_gpu:
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()
        pi_init = None

        if self.init_map_scheme == 'uniform':
            pi_init = None
            
        print("Calculating pi using gcg")

        self.config['cost_mat_path'] = self.cost_mat_path

        if self.lambda_sinkhorn == 'inf':
            pi = np.ones((self.adata_left.n_obs, self.adata_right.n_obs)) / (self.adata_left.n_obs * self.adata_right.n_obs)
            if self.use_gpu:
                pi = torch.from_numpy(pi)
            return pi, -1000
        
        # Try optimal transport with current parameters
        pi, fgw_dist = paste_pairwise_align_modified(
            self.adata_left, self.adata_right,
            alpha=self.alpha, sinkhorn=self.sinkhorn,
            lambda_sinkhorn=self.lambda_sinkhorn,
            dissimilarity=self.dissimilarity, G_init=pi_init,
            numItermax=10000, cost_mat_path=self.cost_mat_path,
            return_obj=True, norm=True, verbose=False,
            backend=backend, use_gpu=self.use_gpu,
            numInnerItermax=self.numInnerIterMax, method='sinkhorn_log'
        )
        
        # Validate pi matrix
        if isinstance(pi, torch.Tensor):
            pi_np = pi.cpu().numpy() if pi.is_cuda else pi.numpy()
        else:
            pi_np = pi
            
        nan_count = np.isnan(pi_np).sum()
        total_elements = pi_np.size
        
        if nan_count > 0:
            nan_percentage = 100 * nan_count / total_elements
            print(f"\n{'='*70}")
            print(f"WARNING: Pi matrix contains {nan_count}/{total_elements} NaN values ({nan_percentage:.2f}%)")
            print(f"This indicates the optimal transport algorithm did not converge properly.")
            print(f"{'='*70}\n")
            
            # If completely failed, try with better parameters
            if nan_count == total_elements:
                print(f"ATTEMPTING AUTOMATIC RECOVERY with improved parameters...")
                print(f"Original: lambda_sinkhorn={self.lambda_sinkhorn}, numInnerItermax={self.numInnerIterMax}")
                
                # Try progressively stronger regularization
                retry_lambdas = [0.01, 0.05, 0.1, 0.5]
                retry_inner_iters = [50000, 100000]
                
                for retry_lambda in retry_lambdas:
                    for retry_inner in retry_inner_iters:
                        print(f"\nRetrying with lambda_sinkhorn={retry_lambda}, numInnerItermax={retry_inner}...")
                        
                        pi_retry, fgw_dist_retry = paste_pairwise_align_modified(
                            self.adata_left, self.adata_right,
                            alpha=self.alpha, sinkhorn=self.sinkhorn,
                            lambda_sinkhorn=retry_lambda,
                            dissimilarity=self.dissimilarity, G_init=pi_init,
                            numItermax=10000, cost_mat_path=self.cost_mat_path,
                            return_obj=True, norm=True, verbose=False,
                            backend=backend, use_gpu=self.use_gpu,
                            numInnerItermax=retry_inner, method='sinkhorn_log'
                        )
                        
                        if isinstance(pi_retry, torch.Tensor):
                            pi_retry_np = pi_retry.cpu().numpy() if pi_retry.is_cuda else pi_retry.numpy()
                        else:
                            pi_retry_np = pi_retry
                        
                        retry_nan_count = np.isnan(pi_retry_np).sum()
                        
                        if retry_nan_count == 0:
                            print(f"\n{'='*70}")
                            print(f"SUCCESS! Convergence achieved with:")
                            print(f"  lambda_sinkhorn={retry_lambda}")
                            print(f"  numInnerItermax={retry_inner}")
                            print(f"Pi matrix is now valid with no NaN values.")
                            print(f"{'='*70}\n")
                            return pi_retry, fgw_dist_retry
                        elif retry_nan_count < total_elements * 0.5:
                            print(f"  Partial improvement: {retry_nan_count}/{total_elements} NaN values ({100*retry_nan_count/total_elements:.2f}%)")
                        else:
                            print(f"  Still failed: {retry_nan_count}/{total_elements} NaN values")
                
                print(f"\n{'='*70}")
                print(f"AUTOMATIC RECOVERY FAILED")
                print(f"Manual intervention required. Please:")
                print(f"  1. Increase lambda_sinkhorn to >= 0.1 in your config")
                print(f"  2. Increase numInnerIterMax to >= 50000")
                print(f"  3. Check input data quality")
                print(f"{'='*70}\n")
        
        return pi, fgw_dist
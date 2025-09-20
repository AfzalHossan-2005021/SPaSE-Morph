import scanpy as sc
import seaborn as sns
import pandas as pd
import numpy as np
from anndata import AnnData
from scipy import sparse

class Preprocessor():
    def __init__(self, config):
        # self.hvg_h5_save_path = config['hvg_h5_savepath']
        pass

    def log_norm(self, adata):
        sc.pp.log1p(adata)

    def hvg(self, adata, top=2000):
        sc.pp.highly_variable_genes(adata, n_top_genes=top)
        return adata[:, adata.var['highly_variable']]

    def pca(self, adata, top=15):
        sc.pp.pca(adata, n_comps=top)
        df_pc = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs.index, columns=[f'pc_{i+1}' for i in range(adata.obsm['X_pca'].shape[1])])
        return df_pc
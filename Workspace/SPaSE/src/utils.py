import os
import ot
import scipy
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm
from anndata import AnnData
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R


def paste_pairwise_align_modified(
        sliceA: AnnData, 
        sliceB: AnnData, 
        alpha: float = 0.1, 
        dissimilarity: str = 'js', 
        sinkhorn: bool = False,
        use_rep: Optional[str] = None,
        lambda_sinkhorn: float = 1, 
        G_init = None, 
        a_distribution = None, 
        b_distribution = None, 
        norm: bool = True, 
        numItermax: int = 10000, 
        backend = ot.backend.NumpyBackend(), 
        use_gpu: bool = False, 
        return_obj: bool = False, 
        verbose: bool = False, 
        gpu_verbose: bool = True,
        cost_mat_path: Optional[str] = None,
        **kwargs) -> Tuple[np.ndarray, Optional[int]]:
        """
        Calculates and returns optimal alignment of two slices. This method is originally from paste module. Modified for this project.
        
        Args:
            sliceA: Slice A to align.
            sliceB: Slice B to align.
            alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
            dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'`` or ``'jensenshannon'``.
            use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
            G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
            a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
            b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
            numItermax: Max number of iterations during FGW-OT.
            norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
            backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
            use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
            return_obj: If ``True``, additionally returns objective function output of FGW-OT.
            verbose: If ``True``, FGW-OT is verbose.
            gpu_verbose: If ``True``, print whether gpu is being used to user.
    
        Returns:
            - Alignment of spots.

            If ``return_obj = True``, additionally returns:
            
            - Objective function output of FGW-OT.
        """

        print("---------------------------------------")
        print('Inside paste_pairwise_align_modified')
        print("---------------------------------------")
        
        # Determine if gpu or cpu is being used
        if use_gpu:                    
            if isinstance(backend,ot.backend.TorchBackend):
                if torch.cuda.is_available():
                    if gpu_verbose:
                        print("gpu is available, using gpu.")
                else:
                    if gpu_verbose:
                        print("gpu is not available, resorting to torch cpu.")
                    use_gpu = False
            else:
                print("We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu.")
                use_gpu = False
        else:
            if gpu_verbose:
                print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")
                
        # subset for common genes
        common_genes = intersect(sliceA.var.index, sliceB.var.index)
        sliceA = sliceA[:, common_genes]
        sliceB = sliceB[:, common_genes]

        # Backend
        nx = backend    
        
        # Calculate spatial distances
        coordinatesA = sliceA.obsm['spatial'].copy()
        coordinatesA = nx.from_numpy(coordinatesA)
        coordinatesB = sliceB.obsm['spatial'].copy()
        coordinatesB = nx.from_numpy(coordinatesB)
        
        if isinstance(nx,ot.backend.TorchBackend):
            coordinatesA = coordinatesA.float()
            coordinatesB = coordinatesB.float()

        D_A = ot.dist(coordinatesA,coordinatesA, metric='euclidean')
        D_B = ot.dist(coordinatesB,coordinatesB, metric='euclidean')

        if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
            D_A = D_A.cuda()
            D_B = D_B.cuda()
        
        # Calculate expression dissimilarity
        A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

        if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
            A_X = A_X.cuda()
            B_X = B_X.cuda()

        if os.path.exists(cost_mat_path):
            print("Loading cost matrix from file system...")
            M_loaded = np.load(cost_mat_path)
            # Validate that the loaded cost matrix has the correct shape
            expected_shape = (sliceA.shape[0], sliceB.shape[0])
            if M_loaded.shape != expected_shape:
                print(f"Loaded cost matrix shape {M_loaded.shape} does not match expected shape {expected_shape}. Regenerating...")
                M = None  # Force regeneration
            else:
                M = M_loaded
        else:
            print("cost_mat_path does not exist.")
            M = None
            
        if M is None:
            if dissimilarity.lower()=='euclidean' or dissimilarity.lower()=='euc':
                M = ot.dist(A_X,B_X)
            elif dissimilarity.lower()=='kl':
                s_A = A_X + 0.01
                s_B = B_X + 0.01
                M = kl_divergence_backend(s_A, s_B)
            elif dissimilarity.lower()=='js' or dissimilarity.lower()=='jensenshannon':
                s_A = A_X + 0.01
                s_B = B_X + 0.01
                M = jensenshannon_divergence_backend(s_A, s_B)
            if cost_mat_path is not None:
                np.save(cost_mat_path, M)
        M = nx.from_numpy(M)

        if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
            M = M.cuda()
        
        # init distributions 
        if a_distribution is None:
            a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
        else:
            a = nx.from_numpy(a_distribution)
            
        if b_distribution is None:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
        else:
            b = nx.from_numpy(b_distribution)

        if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
            a = a.cuda()
            b = b.cuda()
        
        if norm:
            D_A /= nx.min(D_A[D_A>0])
            D_B /= nx.min(D_B[D_B>0])
        
        # Run OT
        if G_init is not None:
            G_init = nx.from_numpy(G_init)
            if isinstance(nx,ot.backend.TorchBackend):
                G_init = G_init.float()
                if use_gpu:
                    G_init.cuda()
            # Validate G_init shape
            expected_g_shape = (sliceA.shape[0], sliceB.shape[0])
            if G_init.shape != expected_g_shape:
                print(f"Warning: G_init has shape {G_init.shape} but expected shape {expected_g_shape}. Using default initialization.")
                G_init = None
        
        assert(sinkhorn == 1)

        pi, logw = my_fused_gromov_wasserstein_gcg(M, D_A, D_B, a, b, lambda_sinkhorn=lambda_sinkhorn, G_init = G_init, loss_fun='square_loss', alpha= alpha, log=True, numItermax=numItermax,verbose=verbose, use_gpu = use_gpu, **kwargs)
        
        pi = nx.to_numpy(pi)
        obj = nx.to_numpy(logw['fgw_dist'])
        if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
            torch.cuda.empty_cache()

        if return_obj:
            return pi, obj
        return pi

def my_fused_gromov_wasserstein_gcg(M, C1, C2, p, q, lambda_sinkhorn=1, G_init = None, loss_fun='square_loss', alpha=0.5, armijo=False, log=False,numItermax=200, use_gpu = False, **kwargs):
        """
        Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
        Also added capability of utilizing different POT backends to speed up computation.
        
        For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
        """
        print("---------------------------------------")
        print("Inside my_fused_gromov_wasserstein_gcg")
        print(f"M shape: {M.shape}")
        print(f"p shape: {p.shape}")
        print(f"q shape: {q.shape}")
        print(f"C1 shape: {C1.shape}")
        print(f"C2 shape: {C2.shape}")
        print("---------------------------------------")

        p, q = ot.utils.list_to_array(p, q)

        p0, q0, C10, C20, M0 = p, q, C1, C2, M

        nx = ot.backend.get_backend(p0, q0, C10, C20, M0)

        # Validate matrix shapes
        n_a, n_b = len(p), len(q)
        if M.shape != (n_a, n_b):
            raise ValueError(f"Cost matrix M has shape {M.shape} but expected shape ({n_a}, {n_b})")
        if C1.shape != (n_a, n_a):
            raise ValueError(f"Distance matrix C1 has shape {C1.shape} but expected shape ({n_a}, {n_a})")
        if C2.shape != (n_b, n_b):
            raise ValueError(f"Distance matrix C2 has shape {C2.shape} but expected shape ({n_b}, {n_b})")

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

        if G_init is None:
            G0 = p[:, None] * q[None, :]
            print(f"G0 initialized with shape: {G0.shape}")
        else:
            G0 = (1/nx.sum(G_init)) * G_init
            print(f"G0 from G_init with shape: {G0.shape}")
        if use_gpu:
            G0 = G0.cuda()

        def f(G):
            return ot.gromov.gwloss(constC, hC1, hC2, G)

        def df(G):
            return ot.gromov.gwggrad(constC, hC1, hC2, G)

        if log:
            print('log true')
            res, log = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=True, **kwargs)
            fgw_dist = log['loss'][-1]
            log['fgw_dist'] = fgw_dist
            return res, log

        else:
            print('log false')
            pi = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=False, **kwargs)
            return pi, -1


def filter_for_common_genes(
    slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/X.sum(axis=1, keepdims=True)
    Y = Y/Y.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

def kl_divergence_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return nx.to_numpy(D)

def kl_divergence_corresponding_backend(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return nx.to_numpy(D)

def jensenshannon_distance_1_vs_many_backend(X, Y, use_gpu: bool = False):
    """
    Returns pairwise Jensenshannon distance (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nx = ot.backend.get_backend(X,Y)
    X = nx.concatenate([X] * Y.shape[0], axis=0)
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    M = (X + Y) / 2.0
    kl_X_M = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl_Y_M = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    if use_gpu and torch.cuda.is_available():
        kl_X_M = kl_X_M.cuda()
        kl_Y_M = kl_Y_M.cuda()
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist

def jensenshannon_divergence_backend(X, Y, use_gpu: bool = False):
    """
    This function is added by Nuwaisir
    
    Returns pairwise JS divergence (over all pairs of samples) of two matrices X and Y.

    Takes advantage of POT backend to speed up computation.

    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)

    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X,Y)

    print(nx.unique(nx.isnan(X)))
    print(nx.unique(nx.isnan(Y)))
        
    
    X = X/nx.sum(X, axis=1, keepdims=True)
    Y = Y/nx.sum(Y, axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y, use_gpu)
        
    print("Finished calculating cost matrix")
    print(nx.unique(nx.isnan(js_dist)))

    if use_gpu:
        return js_dist.cpu().detach().numpy()
    else:
        return js_dist
    # print("vectorized jsd")


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

## Convert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]



def calculate_cost_matrix(adata_left, adata_right, use_gpu: bool = False):
    backend = ot.backend.NumpyBackend()
    if use_gpu:
        backend=ot.backend.TorchBackend()
    nx = backend
    use_rep = None

    common_genes = intersect(adata_left.var.index, adata_right.var.index)
    adata_left = adata_left[:, common_genes]
    adata_right = adata_right[:, common_genes]

    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(adata_left,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(adata_right,use_rep)))
    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()
    s_A = A_X + 0.01
    s_B = B_X + 0.01
    M = kl_divergence_backend(s_A, s_B)
    M = nx.from_numpy(M)
    if use_gpu:
        return M.numpy()
    return M


def rotate(v, angle_deg, center=(0, 0)):
    '''
    v         : numpy array of n 2D points. Shape: (n x 2)
    angle_deg : rotation angle in degrees
    center    : all the points of v will be rotated with respect to center by angle_deg
    '''
    v[:, 0] = v[:, 0] - center[0]
    v[:, 1] = v[:, 1] - center[1]
    rot_mat_2D = R.from_euler('z', angle_deg, degrees=True).as_matrix()[:2, :2]
    v = (rot_mat_2D @ v.T).T
    v[:, 0] = v[:, 0] + center[0]
    v[:, 1] = v[:, 1] + center[1]
    return v


def compute_null_distribution(pi, cost_mat, scheme):
    if scheme == 'all_edges':
        non_zero_idxs_pi = np.nonzero(pi.flatten())[0]
        distances = cost_mat.flatten()[non_zero_idxs_pi]
        weights = pi.flatten()[non_zero_idxs_pi]
    elif scheme == 'left':
        score_mat = pi * cost_mat
        distances = np.sum(score_mat, axis=1) / (1 / pi.shape[0]) * 100
        # print('left', distances.min(), distances.max())
        weights = [1] * len(distances)
    elif scheme == 'right':
        score_mat = pi * cost_mat
        distances = np.sum(score_mat, axis=0) / (1 / pi.shape[1]) * 100
        # print('right', distances.min(), distances.max())
        weights = [1] * len(distances)
    else:
        print("Please set a valid scheme! \n"
              "a) all_edges\n"
              "b) left\n"
              "c) right\n"
              "(at compute_null_distribution function in utils.py)")
        
    return distances, weights


def QC(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)

def scale_coords(adata, key_name):
    adata.obsm[key_name] = adata.obsm[key_name].astype('float')
    x = adata.obsm[key_name][:, 0]
    y = adata.obsm[key_name][:, 1]
    adata.obsm[key_name][:, 0] = x / x.max()
    adata.obsm[key_name][:, 1] = y / y.max()

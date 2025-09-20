import os
import ot
import scipy
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
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
        # print(f'alpha: {alpha}')

        p, q = ot.utils.list_to_array(p, q)

        p0, q0, C10, C20, M0 = p, q, C1, C2, M
        # print('C10')
        # print(C10)
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

        # print('hC1:')
        # print(hC1)

        # print('hC2:')
        # print(hC2)

        def f(G):
            return ot.gromov.gwloss(constC, hC1, hC2, G)

        def df(G):
            return ot.gromov.gwggrad(constC, hC1, hC2, G)

        if log:
            # print('doing gcg')
            # print((1 - alpha) * M)
            print('log true')
            res, log = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=True, **kwargs)
            # res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

            fgw_dist = log['loss'][-1]

            log['fgw_dist'] = fgw_dist
            # log['u'] = log['u']
            # log['v'] = log['v']
            return res, log

        else:
            # return ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
            # print('pi before gcg')
            print('log false')
            pi = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=False, **kwargs)
            # print('pi after gcg')
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

def match_spots_using_spatial_heuristic(
    X,
    Y,
    use_ot: bool = True) -> np.ndarray:
    """
    Calculates and returns a mapping of spots using a spatial heuristic.

    Args:
        X (array-like, optional): Coordinates for spots X.
        Y (array-like, optional): Coordinates for spots Y.
        use_ot: If ``True``, use optimal transport ``ot.emd()`` to calculate mapping. Otherwise, use Scipy's ``min_weight_full_bipartite_matching()`` algorithm.

    Returns:
        Mapping of spots using a spatial heuristic.
    """
    n1,n2=len(X),len(Y)
    X,Y = norm_and_center_coordinates(X),norm_and_center_coordinates(Y)
    dist = scipy.spatial.distance_matrix(X,Y)
    if use_ot:
        pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    else:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(scipy.sparse.csr_matrix(dist))
        pi = np.zeros((n1,n2))
        pi[row_ind, col_ind] = 1/max(n1,n2)
        if n1<n2: pi[:, [(j not in col_ind) for j in range(n2)]] = 1/(n1*n2)
        elif n2<n1: pi[[(i not in row_ind) for i in range(n1)], :] = 1/(n1*n2)
    return pi

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

def jensenshannon_distance_1_vs_many_backend(X, Y):
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
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist

def jensenshannon_divergence_backend(X, Y):
    """
    This function is added ny Nuwaisir
    
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
    # nx = ot.backend.NumpyBackend()

    # X = X.cpu().detach().numpy()
    # Y = Y.cpu().detach().numpy()

    print(nx.unique(nx.isnan(X)))
    print(nx.unique(nx.isnan(Y)))
        
    
    X = X/nx.sum(X, axis=1, keepdims=True)
    Y = Y/nx.sum(Y, axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y)
        
    print("Finished calculating cost matrix")
    print(nx.unique(nx.isnan(js_dist)))

    if torch.cuda.is_available():
        return js_dist.cpu().detach().numpy()
    else:
        return js_dist
    # print("vectorized jsd")
    # X = X/nx.sum(X,axis=1, keepdims=True)
    # Y = Y/nx.sum(Y,axis=1, keepdims=True)

    # mid = (X[:, None] + Y) / 2
    # n = X.shape[0]
    # m = Y.shape[0]
    # d = X.shape[1]
    # l = nx.ones((n, m, d)) * X[:, None, :]
    # r = nx.ones((n, m, d)) * Y[None, :, :]
    # l_2d = nx.reshape(l, (-1, l.shape[2]))
    # r_2d = nx.reshape(r, (-1, r.shape[2]))
    # m_2d = (l_2d + r_2d) / 2.0
    # kl_l_m = kl_divergence_corresponding_backend(l_2d, m_2d)
    # kl_r_m = kl_divergence_corresponding_backend(r_2d, m_2d)

    # js_dist = nx.sqrt((kl_l_m + kl_r_m) / 2.0)
    # return nx.reshape(js_dist, (n, m))


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

def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))

def apply_trsf(
    M: np.ndarray,
    translation: List[float],
    points: np.ndarray) -> np.ndarray:
    """
    Apply a rotation from a 2x2 rotation matrix `M` together with
    a translation from a translation vector of length 2 `translation` to a list of
    `points`.

    Args:
        M (nd.array): a 2x2 rotation matrix.
        translation (nd.array): a translation vector of length 2.
        points (nd.array): a nx2 array of `n` points 2D positions.

    Returns:
        (nd.array) a nx2 matrix of the `n` points transformed.
    """
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation)
    trsf = np.identity(3)
    trsf[:-1, :-1] = M
    tr = np.identity(3)
    tr[:-1, -1] = -translation
    trsf = trsf @ tr

    flo = points.T
    flo_pad = np.pad(flo, ((0, 1), (0, 0)), constant_values=1)
    return ((trsf @ flo_pad)[:-1]).T

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]


def get_random_pi_init(n, m, scheme):
    if scheme == 'one_to_one':
        # if n < m:
        #     idx = list(range(n))
        #     random.shuffle(idx)
        #     pi_init = np.zeros((n, m))
        #     pi_init[idx, :] = 1 / n
        # else:
        pass
    else:
        print('get_random_pi_init\'s scheme doesn\'t match')


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    idx = largest_indices(G, top)
    for l in range(len(idx[0])):
        plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]],
                 alpha=alpha*(1-weight_alpha)+(weight_alpha*G[idx[0][l], idx[1][l]] / mx), c='k')


def plot2D_samples_mat_modified(xs, xt, G, labels=None, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, **kwargs):
    xs_xmax, xs_xmin = max(xs[:, 0]), min(xs[:, 0])
    xt_xmax, xt_xmin = max(xt[:, 0]), min(xt[:, 0])

    xs_ymax, xs_ymin = max(xs[:, 1]), min(xs[:, 1])
    xt_ymax, xt_ymin = max(xt[:, 1]), min(xt[:, 1])

    xs_total = len(xs)
    xt_total = len(xt)

    threshold_1 = 0.30
    threshold_2 = 0.50
    threshold_3 = 0.60
    threshold_4 = 0.80
    threshold_5 = 1

    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()

    idx = largest_indices(G, top)

    xs_spot_colors = ['gray'] * xs_total
    xt_spot_colors = ['gray'] * xt_total
    for l in range(len(idx[0])):
        rgb = np.random.rand(3,)
        ax_perc = (xs[idx[0][l], 1] - xs_ymin) / (xs_ymax - xs_ymin)  # for y
        # ax_perc = (xs[idx[0][l], 0] - xs_xmin) / (xs_xmax - xs_xmin) # for x

        spot_color = 'gray'
        if ax_perc < threshold_1:
            spot_color = 'orange'
        elif ax_perc < threshold_2:
            spot_color = 'red'
        elif ax_perc < threshold_3:
            spot_color = 'green'
        elif ax_perc < threshold_4:
            spot_color = 'purple'
        else:
            spot_color = 'blue'
        # plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]], linewidth=1, alpha=0.1+alpha*(1-weight_alpha)+(weight_alpha*G[idx[0][l],idx[1][l]] /mx),c='black')

        xs_spot_colors[idx[0][l]] = spot_color
        xt_spot_colors[idx[1][l]] = spot_color

    return xs_spot_colors, xt_spot_colors


def plot_slice_pairwise_alignment(slice1, slice2, pi, thr=1-1e-8, alpha=0.05, top=1000, name='', save=False, weight_alpha=False):
    coordinates1, coordinates2 = slice1.obsm['spatial'], slice2.obsm['spatial']
    offset = (coordinates1[:, 0].max()-coordinates2[:, 0].min())*1.1
    temp = np.zeros(coordinates2.shape)
    temp[:, 0] = offset
    plt.figure(figsize=(20, 10))
    plot2D_samples_mat(coordinates1, coordinates2+temp, pi, thr=thr,
                       c='k', alpha=alpha, top=top, weight_alpha=weight_alpha)
    plt.scatter(coordinates1[:, 0], coordinates1[:, 1], linewidth=0, s=100, marker=".", color=list(slice1.obs['layer_guess_reordered'].map(
        dict(zip(slice1.obs['layer_guess_reordered'].cat.categories, slice1.uns['layer_guess_reordered_colors'])))))
    plt.scatter(coordinates2[:, 0]+offset, coordinates2[:, 1], linewidth=0, s=100, marker=".", color=list(slice2.obs['layer_guess_reordered'].map(
        dict(zip(slice2.obs['layer_guess_reordered'].cat.categories, slice2.uns['layer_guess_reordered_colors'])))))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()


def plot_slice_pairwise_alignment_modified(slice1, slice2, pi, thr=1-1e-8, alpha=0.05, top=1000, name='', save=False, weight_alpha=False, use_max=False, save_dir='', show=False, invert_y=True, invert_x=False):
    coordinates1, coordinates2 = slice1.obsm['spatial'], slice2.obsm['spatial']
    offset = (coordinates1[:, 0].max()-coordinates2[:, 0].min())*1.1 + 30
    temp = np.zeros(coordinates2.shape)
    temp[:, 0] = offset
    plt.figure(figsize=(12, 6))
    c1, c2 = plot2D_samples_mat_modified(coordinates1, coordinates2+temp, pi, thr=thr,
                                         c='k', alpha=alpha, top=top, weight_alpha=weight_alpha, use_max=use_max)

    plt.scatter(coordinates1[:, 0], coordinates1[:, 1],
                linewidth=0, s=100, marker=".", c=c1)
    plt.scatter(coordinates2[:, 0]+offset, coordinates2[:,
                1], linewidth=0, s=100, marker=".", c=c2)
    if invert_y:
        plt.gca().invert_yaxis()
    if invert_x:
        plt.gca().invert_xaxis()
    
    
    plt.axis('off')
    if save:
        if save_dir == '':
            print('Save failed! save_dir is not an empty string!')
            return
        print(f'saving at: {save_dir}/{name}')
        plt.savefig(f'{save_dir}/{name}')
    if show:
        plt.show()
    plt.close()


def calculate_cost_matrix(adata_left, adata_right):
    use_gpu = torch.cuda.is_available()
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


def badness_of_mapping(tissue_left, tissue_right, angle):
    '''
    tissue_left: dict object having (x, y) as keys and labels as values
    tissue_right: dict object having (x, y) as keys and labels as values
    angle   : Rotation angle of tissue_right in degrees
    '''
    coor_left = np.array(list(tissue_left.keys()))
    coor_right = np.array(list(tissue_right.keys()))

    labels_left = np.array(list(tissue_left.values()))
    labels_right = np.array(list(tissue_right.values()))

    coor_left = norm_and_center_coordinates(coor_left)
    coor_right = norm_and_center_coordinates(coor_right)

    coor_right = rotate(coor_right, angle)

    # plt.figure(figsize = (20, 10))
    # plt.scatter(coor_left[:, 0], coor_left[:, 1], c = labels_left)
    # plt.scatter(coor_right[:, 0] + 70, coor_right[:, 1], c = labels_right)

    label_count_left = len(np.unique(labels_left))
    label_count_right = len(np.unique(labels_right))

    if label_count_left != label_count_right:
        warnings.warn("Missmatched label count")

    label_to_coor_left = {}
    label_to_coor_right = {}

    # print(np.unique(labels_left))

    label_to_coor_left[1] = np.array(coor_left[np.where(labels_left == 0)[0]])
    label_to_coor_left[2] = np.array(coor_left[np.where(labels_left == 1)[0]])

    # print(label_to_coor_left[1])
    # print(label_to_coor_left[2])

    label_to_coor_right[1] = np.array(
        coor_right[np.where(labels_right == 0)[0]])
    label_to_coor_right[2] = np.array(
        coor_right[np.where(labels_right == 1)[0]])

    # for i in range(len(labels_left)):
    #     label = labels_left[i]
    #     if label not in label_to_coor_left:
    #         label_to_coor_left[label] = []
    #     label_to_coor_left[label].append(list(coor_left[i]))

    # for i in range(len(labels_right)):
    #     label = labels_right[i]
    #     if label not in label_to_coor_right:
    #         label_to_coor_right[label] = []
    #     label_to_coor_right[label].append(list(coor_right[i]))

    # label_to_coor_left[1] = np.array(label_to_coor_left[1])
    # label_to_coor_left[2] = np.array(label_to_coor_left[2])
    # label_to_coor_right[1] = np.array(label_to_coor_right[1])
    # label_to_coor_right[2] = np.array(label_to_coor_right[2])

    cnt_left_1 = len(label_to_coor_left[1])
    cnt_left_2 = len(label_to_coor_left[2])
    cnt_right_1 = len(label_to_coor_right[1])
    cnt_right_2 = len(label_to_coor_right[2])

    label_map = {
        1: 2,
        2: 1,
    }

    # print(cnt1)

    if((cnt_left_1 < cnt_left_2 and cnt_right_1 < cnt_right_2) or (cnt_left_1 > cnt_left_2 and cnt_right_1 > cnt_right_2)):
        label_map = {
            1: 1,
            2: 2,
        }

    kdtree_right_1 = scipy.spatial.KDTree(label_to_coor_right[1])
    kdtree_right_2 = scipy.spatial.KDTree(label_to_coor_right[2])

    kdtrees_right = {
        1: kdtree_right_1,
        2: kdtree_right_2,
    }

    badness = 0

    for label in label_to_coor_left:
        target_label = label_map[label]
        for spot in label_to_coor_left[label]:
            nearest_spot_idx = kdtrees_right[target_label].query(spot)[1]
            nearest_spot = label_to_coor_right[target_label][nearest_spot_idx]
            badness += euc_dist(spot, nearest_spot)
    return badness


def find_nearest_point(point_list, target_point):
    idx = scipy.spatial.KDTree(point_list).query(target_point)[1]
    return idx, point_list[idx]


def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))


def get_2_segment_spagcn(sample):
    return pd.read_csv(f'/home/nuwaisir/Corridor/MSC/Network_science/SpaGCN/tutorial/data/2_segments/{sample}_2_segments.csv', index_col=0)


def get_coor_and_label(sample):
    df_2_seg = get_2_segment_spagcn(sample)
    coor = df_2_seg[['x_array', 'y_array']].values
    label = df_2_seg['refined_pred'].values
    return coor, label


def get_adata(sample, angle=0):
    h5_path = f"/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg/Data/Human_DLPFC/{sample}/reading_h5/"
    h5_file = f'{sample}_filtered_feature_bc_matrix.h5'
    adata = sc.read_visium(path=h5_path, count_file=h5_file)
    adata.var_names_make_unique()
    adata.obsm['spatial'] = rotate(adata.obsm['spatial'], angle)
    # adata = read_10x_h5(f"/home/nuwaisir/Corridor/Thesis_ug/ScribbleSeg/Data/Human_DLPFC/{sample}/reading_h5/{sample}_filtered_feature_bc_matrix.h5")
    return adata


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

def mirror(coor):
    coor_ret = np.zeros((coor.shape[0], coor.shape[1]))
    coor_ret[:, 0] = coor[:, 0]
    coor_ret[:, 1] = -coor[:, 1]
    return coor_ret

def euc_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def make_format(coor, labels):
    d = {}
    for i in range(len(coor)):
        d[(coor[i][0], coor[i][1])] = labels[i]
    return d

def find_init_angle_of_rotation(sample_left, sample_right, adata_left, adata_right):


    # TO ADD: Run spagcn if the 2-segment outputs are not present


    df_2_seg_left = get_2_segment_spagcn(sample_left)
    df_2_seg_right = get_2_segment_spagcn(sample_right)

    adata_left.obs.index = trim_barcode(adata_left.obs.index)
    adata_right.obs.index = trim_barcode(adata_right.obs.index)

    adata_left.obs = adata_left.obs.join(df_2_seg_left[['refined_pred']])
    adata_right.obs = adata_right.obs.join(df_2_seg_right[['refined_pred']])

    coor_1 = adata_left.obsm['spatial']
    coor_2 = adata_right.obsm['spatial']
    label_1 = adata_left.obs['refined_pred']
    label_2 = adata_right.obs['refined_pred']
    dict_coor_label_1 = make_format(coor_1, label_1)
    dict_coor_label_2 = make_format(coor_2, label_2)


    # Maybe there is some way to improve this method except searching exhaustively

    target_angle = 0
    mn_badness = 100000000000000000000
    for angle in range(0, 360, 5):
        print(angle, end=', ')
        score = badness_of_mapping(dict_coor_label_1, dict_coor_label_2, angle)
        if mn_badness > score:
            mn_badness = score
            target_angle = angle
        # badness_to_angle[score] = angle
    # print('\ntarget_angle', -target_angle)


    return -target_angle

def trim_barcode(v):
    return list(map(lambda x: x.split('-')[0]+'-1', v))

def preprocess_adata_for_seg_and_rot(adata):
    df_2_seg = get_2_segment_spagcn(adata)
    adata.obs.index = trim_barcode(adata.obs.index)
    adata.obs = adata.obs.join(df_2_seg[['refined_pred']])
    coor_1 = adata.obsm['spatial']
    label_1 = adata.obs['refined_pred']
    coor_label_dict = make_format(coor_1, label_1)

def remove_all_same_rows_and_cols(grid_idx):
    rows_to_be_kept = []
    cols_to_be_kept = []

    for i in range(len(grid_idx)):
        if np.all(grid_idx[i] == grid_idx[i][0]):
            rows_to_be_kept.append(False)
        else: rows_to_be_kept.append(True)

    for i in range(len(grid_idx[0])):
        if np.all(grid_idx[:,i] == grid_idx[:,i][0]):
            cols_to_be_kept.append(False)
        else: cols_to_be_kept.append(True)

    return grid_idx[rows_to_be_kept][:, cols_to_be_kept]

def get_2hop_adatas(adata):
    n = adata.obs['array_row'].max() + 1
    m = adata.obs['array_col'].max() + 1
    barcode_grid = np.empty([n, m], dtype='<U100')
    grid_idx = np.zeros((n, m)) - 1
    spot_rows = adata.obs['array_row']
    spot_cols = adata.obs['array_col']
    barcode_grid[spot_rows, spot_cols] = adata.obs.index
    grid_idx[spot_rows, spot_cols] = range(len(adata.obs.index))
    barcode_index = dict(zip(adata.obs.index, range(len(adata.obs.index))))
    # remove_all_same_rows_and_cols(grid_idx)

    col_max = grid_idx.max(axis=0)
    col_idxs = np.argwhere(col_max != -1).reshape(-1)

    
    grid_idx_0 = grid_idx[:, col_idxs[::2]]
    grid_idx_1 = grid_idx[:, col_idxs[1::2]]

    idxs_0_2hop = grid_idx_0[grid_idx_0 != -1].astype('int')
    idxs_1_2hop = grid_idx_1[grid_idx_1 != -1].astype('int')
    adata_0 = adata[idxs_0_2hop]
    adata_1 = adata[idxs_1_2hop]
    return adata_0, adata_1

# def apply_PASTE(adata_left, adata_right, alpha=0.1):
#     # alpha = 0.1
#     try:
#         pi = pst.pairwise_align(adata_left, adata_right,alpha=alpha,G_init=None,numItermax=10000,verbose=False,backend=ot.backend.TorchBackend(),use_gpu=torch.cuda.is_available())
#     except:
#         pi = pst.pairwise_align(adata_left, adata_right,alpha=alpha,G_init=None,numItermax=10000,verbose=False)
#     return pi

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
        print("Please set a valid scheme! \n\
            a) all_edges\n\
            b) left\n\
            c) right\n\
            (at compute_null_distribution function in utils.py)")
        
    return distances, weights


def QC(adata):
    # sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :]
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

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

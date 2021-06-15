import numpy as np
import pandas as pd
from scipy.stats import mode

def compute_distance(p1, p2, distance_function=None):
    if distance_function is None:
        distance_function = lambda x1, x2: np.abs((x2-x1)**2)
    ret = distance_function(p1, p2)
    ret = ret.sum(axis=-1)
    ret = np.abs(ret)
    return ret


def prepare_data(data, numerical=[0,1], nominal=[2]):
    data_num = data.iloc[:,numerical].copy()
    data_nom = data.iloc[:,nominal].copy()
    
    for ii in nominal:
        to_replace = np.sort(data.iloc[:,ii].unique())
        to_replace = dict(zip(to_replace, np.arange(to_replace.shape[0])))
        data_nom.loc[:,ii] = data_nom.loc[:,ii].replace(to_replace, None)
    
    return data_num.to_numpy(), data_nom.to_numpy()


def k_means(data, numerical=[0,1], nominal=[2], K=3, iters=1, method=None, distance_function_num=None, distance_function_nom=None, mode_nominal=None):
    # prepare initial methods and data
    if method == 'sorted':
        distance_function_num = lambda x1, x2: np.abs((x2-x1)**2)
        distance_function_nom = lambda x1, x2: np.abs((x2-x1))
        mode_nominal = False
    elif method == 'hamming':
        distance_function_num = lambda x1, x2: np.abs((x2-x1)**2)
        distance_function_nom = lambda x1, x2: x1!=x2
        mode_nominal = True
    elif method == 'gower':
        def gower(x1, x2):
            ret = np.abs((x2-x1))/(np.ptp(x1, axis=1)[0])
            return ret
        distance_function_num = lambda x1, x2: np.abs((x2-x1))/(np.ptp(x1, axis=1)[0])
        distance_function_nom = lambda x1, x2: x1!=x2
    else:
        if len(nominal)<1:
            mode_nominal = False
        print('Applying custom distance function')
    N = data.shape[0]
    data_num, data_nom = prepare_data(data, numerical, nominal)
    
    # choose initial clusters
    rnd = np.random.choice(N, size=K, replace=False)
    cluster_num, cluster_nom = data_num[rnd], data_nom[rnd]
    
    # prepare to return 
    ret_cluster = []
    ret_idx = []
    
    # make a matrix of points - to vectorize and speed-up the whole process
    data_num, data_nom = np.array([data_num]*K), np.array([data_nom]*K)
    
    # loop for computing new cluster
    for _ in range(iters):
        # make and modify matrix of cluster
        cluster_num, cluster_nom = np.array([cluster_num]*N), np.array([cluster_nom]*N)
        cluster_num, cluster_nom = np.swapaxes(cluster_num, 0, 1), np.swapaxes(cluster_nom, 0, 1)
        
        # compute distances
        distance_num = compute_distance(data_num, cluster_num, distance_function_num)
        distance_nom = compute_distance(data_nom, cluster_nom, distance_function_nom)
        distance = distance_num + distance_nom
        
        # determine to which cluster points belong (and pass it to return)
        idx = distance.argmin(axis=0)
        
        #check if the numbers of cluster is constant
        if np.unique(idx).shape[0]!=K: break
        ret_idx += [idx.tolist()]
        
        # make and modify matrix of idx (belonges points to clusters)
        idx = np.array([idx]*K).T
        compares = idx == np.arange(K)
        divisor = compares.sum(axis=0)

        # final computing to position of clusters - the case of numerical
        cluster_num, cluster_nom = np.swapaxes(data_num, 0, -1) * compares, np.swapaxes(data_nom, 0, -1) * compares
        cluster_num, cluster_nom = cluster_num.sum(axis=1)/divisor, cluster_nom.sum(axis=1)/divisor
        cluster_num, cluster_nom = np.swapaxes(cluster_num, 0, -1),  np.swapaxes(cluster_nom, 0, -1)
        
        # in the case of using mode instead of mean for nominal values
        if mode_nominal:
            idx = np.array(ret_idx[-1])
            cluster_nom = []
            for ii in range(K):
                cluster_nom += [mode(data_nom[0,idx==ii])[0][0]]
            cluster_nom = np.array(cluster_nom)
        
        # check if centers are changing
        if _ >0:
            if np.append(cluster_num, cluster_nom, axis=1).tolist() == ret_cluster[-1]:
                break
        ret_cluster += [np.append(cluster_num, cluster_nom, axis=1).tolist()]
    return np.array(ret_cluster), np.array(ret_idx)
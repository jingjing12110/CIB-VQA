# @File  :over_smoothing_metrics.py
# @Desc  :https://github.com/Kaixiong-Zhou/DGN.git
import numpy as np


# **************************************************
# Metric 1: Group Distance Ratio
# **************************************************
def dis_cluster(X, Y, num_classes):
    X_labels = []
    for i in range(num_classes):
        X_label = X[Y == i].data.cpu().numpy()
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)
    
    dis_intra = 0.
    for i in range(num_classes):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= num_classes
    
    dis_inter = 0.
    for i in range(num_classes - 1):
        for j in range(i + 1, num_classes):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(num_classes * (num_classes - 1) / 2)
    dis_inter /= num_inter
    
    return dis_intra, dis_inter


def compute_gdr(X, Y, num_classes, gap='close'):
    """compute the intra-group and inter-group distances first
    then obtain the group distance ratio
    @param X:
    @param Y:
    @param num_classes:
    @param gap:
    @return:
    """
    dis_intra, dis_inter = dis_cluster(X, Y, num_classes)
    if gap == 'close':
        # if the intra-group and inter-group distances are close,
        # we assign them the same values and have the distance ratio of 1.
        distance_gap = dis_inter - dis_intra
        dis_ratio = 1. if distance_gap < 0.35 else dis_inter / dis_intra
    else:
        dis_ratio = dis_inter / dis_intra
    # if both dis_inter and dis_intra are close to zero,
    # the value of dis_ratio is nan
    # in this case, we assign the distance ratio to 1.
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    return dis_ratio


# **************************************************
# Metric 2: Instance Information Gain
# **************************************************
def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False,
                                                 return_inverse=True,
                                                 return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def Kget_dists(X):
    """computing the pairwise distance matrix for a set of vectors specified
    by the matrix X.
    """
    x2 = np.sum(np.square(X), axis=1, keepdims=True)
    dists = x2 + x2.T - 2 * np.matmul(X, X.T)
    return dists


def entropy_estimator_kl(x, var):
    dims, N = float(x.shape[1]), float(x.shape[0])
    dists = Kget_dists(x)
    dists2 = dists / (2 * var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
    h = -np.mean(lprobs)
    
    return dims / 2 + h


def entropy_estimator_bd(x, var):
    """Bhattacharyya-based lower bound on entropy of mixture of Gaussian with
     covariance matrix var * I
    Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances,
    Entropy, 2017. Section 4.
    @param x:
    @param var:
    @return:
    """
    dims, N = float(x.shape[1]), float(x.shape[0])
    val = entropy_estimator_kl(x, 4 * var)
    return val + np.log(0.25) * dims / 2


def kde_condentropy(x, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = x.shape[1]
    return (dims / 2.0) * (np.log(2 * np.pi * var) + 1)


def mi_kde(h, inputdata, var=0.1):
    """compute the mutual information between the input and
    the final representation
    @param h: hidden representation at the final layer
    @param inputdata: the input attribute matrix X
    @param var: noise variance used in estimate the mutual information in KDE
    @return:
    """
    nats2bits = float(1.0 / np.log(2))
    h_norm = np.sum(np.square(h), axis=1, keepdims=True)
    h_norm[h_norm == 0.] = 1e-3
    h = h / np.sqrt(h_norm)
    input_norm = np.sum(np.square(inputdata), axis=1, keepdims=True)
    input_norm[input_norm == 0.] = 1e-3
    inputdata = inputdata / np.sqrt(input_norm)
    
    # the entropy of the input
    entropy_input = entropy_estimator_bd(inputdata, var)
    
    # compute the entropy of input
    # given the hidden representation at the final layer
    entropy_input_h = 0.
    indices = np.argmax(h, axis=1)
    indices = np.expand_dims(indices, axis=1)
    p_h, unique_inverse_h = get_unique_probs(indices)
    p_h = np.asarray(p_h).T
    for i in range(len(p_h)):
        labelixs = unique_inverse_h == i
        entropy_input_h += p_h[i] * entropy_estimator_bd(
            inputdata[labelixs, :], var)
    
    # the mutual information between the input
    # and the hidden representation at the final layer
    MI_HX = entropy_input - entropy_input_h
    
    return nats2bits * MI_HX

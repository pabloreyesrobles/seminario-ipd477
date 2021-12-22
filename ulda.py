import numpy as np

def ulda(data, classes):

    m, n = data.shape
    cc = np.mean(data, axis=0)
    k = np.max(classes) + 1

    B = np.zeros([m + k, n])

    for i in range(k):
        loc = classes == i
        num = np.sum(loc)
        tmp = np.mean(data[loc], axis=0)

        B[i, :] = np.sqrt(num) * (tmp - cc)
        B[np.where(loc) + k, :] = data[loc] - tmp

    Hb = B[:k, :].T
    size_low = np.linalg.matrix_rank(Hb)

    Ht = data - cc
    Ht = Ht.T

    U1, D1, _ = np.linalg.svd(Ht)

    s = np.linalg.matrix_rank(Ht)
    D1 = np.diag(D1)[:s, :s]
    U1 = U1[:, :s]

    d1 = np.diag(D1)
    d1 = d1 / d1 # ?
    D1 = np.diag(d1)

    B = np.matmul(np.matmul(D1, U1.T), Hb)

    P, _, _ = np.linalg.svd(B)
    X = np.matmul(np.matmul(U1, D1), P)
    return X[:, :size_low]

def ulda_feature_reduction(data, classes, n_feat):
    G = ulda(data, classes)
    features = np.matmul(data, G)
    n_feat_ulda = features.shape[1]
    n_feat_out = np.min([n_feat, n_feat_ulda])

    return features[:, :n_feat_out]

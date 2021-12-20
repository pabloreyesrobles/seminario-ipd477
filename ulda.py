import numpy as np
from numpy.core.fromnumeric import size

def ulda(data, classes):

    m, n = data.shape
    cc = np.mean(data, axis=0)
    k = np.max(classes)

    B = np.zeros([m + k, n])

    for i in range(k):
        loc = classes == i
        num = np.sum(loc)
        tmp = np.mean(data[loc], axis=0)

        B[i, :] = np.sqrt(num) * (tmp - cc)
        B[np.where(loc) + k, :] = data[loc] - tmp

    Hb = B[:k, :]
    size_low = np.linalg.matrix_rank(Hb)

    Ht = data - cc

    U1, D1, _ = np.linalg.svd(Ht)

    s = np.linalg.matrix_rank(Ht)
    D1 = D1[:s, :s]
    U1 = U1[:, :s]

    d1 = np.diag(D1)
    d1 = d1 / d1 # ?
    D1 = np.diag(d1)

    B = D1 * U1 * Hb

    P, _, _ = np.linalg.matrix_rank(B)
    X = U1 * D1 * P
    return X[:, :size_low]


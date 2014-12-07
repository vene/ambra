import numpy as np
from sklearn.utils.validation import check_array, check_random_state

def pairwise_transform(X, Y):
    """Form comparable pairs with interval-annotated entries.

    Parameters
    ----------

    X: array-like, shape (n_samples x n_features)
        The feature representation of the instances.

    Y: array_like, shape (n_samples x 2)
        The lower and upper bounds of the interval of each instance.

    """
    X = check_array(X, accept_sparse=None)
    Y = check_array(Y, accept_sparse=None)
    if Y.shape[1] != 2:
        raise ValueError("Y must have two columns, represeting the lower "
                         "and upper bound of the interval for each entry.")

    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    chunks = []
    chunk_idx = []
    for k, (x, (y_min, y_max)) in enumerate(zip(X, Y)):
        X_rest, Y_rest = X[1 + k:], Y[1 + k:]
        idx_rest = idx[1 + k:]
        before = Y_rest[:, 1] < y_min
        after = Y_rest[:, 0] > y_max
        if np.sum(before):
            chunks.append(X_rest[before] - x)
            chunk_idx.append(np.array([(i, k) for i in idx_rest[before]]))
        if np.sum(after):
            chunks.append(x - X_rest[after])
            chunk_idx.append(np.array([(k, i) for i in idx_rest[after]]))

    if len(chunks):
        return np.row_stack(chunks), np.row_stack(chunk_idx)
    else:
        raise ValueError("Empty slice: no pairs can be formed.")
        # return X[:0].copy(), np.array([[]])  # fail silently

def flip_pairs(X_pairwise, random_state=None):
    rng = check_random_state(random_state)
    n_pairs = X_pairwise.shape[0]
    y = np.ones(n_pairs)
    flip = rng.choice(range(n_pairs), size=n_pairs / 2, replace=False)
    y[flip] = -1
    X_flipped = X_pairwise * y[:, np.newaxis]

    return X_flipped, y


if __name__ == '__main__':
    X = np.arange(6)[:, np.newaxis]
    Y = [[4, 7], [1, 3], [2, 4], [8, 15], [5, 6], [1, 2]]

    X_pw, res = pairwise_transform(X, Y)
    X_pw, y_pw = flip_pairs(X_pw, random_state=0)
    for x, y, i in zip(X_pw, y_pw, res):
        print x, y, i

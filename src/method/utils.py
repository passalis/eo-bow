import numpy as np
import theano.tensor as T
import theano

floatX = theano.config.floatX


def get_class_mapping(labels):
    """
    Returns the pi_ij matrix (1: if the i-th object belongs to class j), 0 otherwise)
    :param labels the labels
    :return the pi (label to instance mapping) matrix
    """
    unique = np.unique(labels)
    mapping = np.zeros((labels.shape[0], unique.shape[0]))
    for i in range(labels.shape[0]):
        idx = np.where(labels[i] == unique)
        mapping[i, idx] = 1

    return mapping


def symbolic_distance_matrix(A, B):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    :param A:
    :param B:
    :return:
    """
    aa = T.sum(A * A, axis=1)
    bb = T.sum(B * B, axis=1)
    AB = T.dot(A, T.transpose(B))

    AA = T.transpose(T.tile(aa, (bb.shape[0], 1)))
    BB = T.tile(bb, (aa.shape[0], 1))

    D = AA + BB - 2 * AB
    # D = T.fill_diagonal(D, 0)
    D = T.maximum(D, 0)
    D = T.sqrt(D)
    return D


def subsample(X, n_samples=100):
    """
    Subsamples n_samples feature vectors from each object
    """
    objects = []

    for x in X:
        features = x

        # If less than n_samples features exist, repeat some of them
        if x.shape[0] < n_samples:
            n_repeats = np.int64(np.ceil(n_samples / x.shape[0]))
            for i in range(n_repeats):
                features = np.vstack((features, x))

        # Get a sample of the permutation of the features
        idx = np.random.permutation(features.shape[0])
        idx = idx[:n_samples]
        objects.append(features[idx, :])
    return np.asarray(objects, dtype=floatX)

import theano
import numpy as np
import theano.tensor as T
from utils import symbolic_distance_matrix, get_class_mapping

floatX = theano.config.floatX


class SoftEntropy:
    """
    Defines the symbolic calculation of the soft entropy
    """

    def __init__(self, labels, m=0.1):
        """
        Initializes the Soft Entropy Class
        """
        self.m = m

        # Histograms
        self.S = T.matrix(name='S', dtype=theano.config.floatX)

        # Entropy centers
        self.C = theano.shared(np.asarray(np.zeros((1, 1)), dtype=floatX), name='C')
        self.labels = labels
        self.mapping = theano.shared(np.asarray(get_class_mapping(labels), dtype=floatX), name='mapping')

        # Compile functions
        self.calculate_soft_entropy = theano.function([self.S], self._sym_entropy(self.S))

    def init_centers(self, S):
        """
        Gets the histograms S and positions one center above each class
        """
        labels = self.labels
        unique_labels = np.unique(labels)
        centers = None

        for label in unique_labels:
            idx = np.squeeze(labels == label)
            cur_S = S[idx, :]
            cur_center = np.mean(cur_S, axis=0)
            if centers is None:
                centers = cur_center
            else:
                centers = np.vstack((centers, cur_center))
        centers = np.asarray(centers, dtype=floatX)
        self.C.set_value(centers)

    def _sym_entropy(self, S):
        """
        Defines the symbolic calculation of the soft entropy
        """

        distances = symbolic_distance_matrix(S, self.C)
        Q = T.nnet.softmax(-distances / self.m)

        # Calculates the fuzzy membership vector for each histogram S
        Nk = T.sum(Q, axis=0)

        H = T.dot(self.mapping.T, Q)
        P = H / Nk

        entropy_per_cluster = P * T.log2(P)
        entropy_per_cluster = T.switch(T.isnan(entropy_per_cluster), 0, entropy_per_cluster)
        entropy_per_cluster = entropy_per_cluster.sum(axis=0)

        Rk = Nk / Nk.sum()
        E = -(entropy_per_cluster * Rk).sum()
        return T.squeeze(E)

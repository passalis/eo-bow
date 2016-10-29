
import theano
import numpy as np
import numpy.linalg as lin
from sklearn.neighbors import NearestNeighbors
import theano.tensor as T

floatX = theano.config.floatX


class SoftEntropy:
    """
    Defines the symbolic calculation of the soft entropy
    Also, provides numpy-based implementations for calculating the soft and the hard entropy
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
        self.calculate_soft_entropy = theano.function([self.S], self.sym_entropy(self.S))


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

    # def sym_get_similarity(self, x):
    #    """
    #    Calculates the fuzzy membership vector for each vector x
    #    """
    #    similarity = T.nnet.softmax(-(x - self.C).norm(2, axis=1) / self.m)
    #    return T.squeeze(similarity)

    def sym_entropy(self, S):
        """
        Defines the symbolic calculation of the soft entropy
        """

        distances = symbolic_distance_matrix(S, self.C)
        Q = T.nnet.softmax(-distances / self.m)

        # Calculates the fuzzy membership vector for each histogram S
        # Q, scan_u = theano.map(fn=self.sym_get_similarity, sequences=[S])

        Nk = T.sum(Q, axis=0)

        H = T.dot(self.mapping.T, Q)
        P = H / Nk

        entropy_per_cluster = P * T.log2(P)
        entropy_per_cluster = T.switch(T.isnan(entropy_per_cluster), 0, entropy_per_cluster)
        entropy_per_cluster = entropy_per_cluster.sum(axis=0)

        Rk = Nk / Nk.sum()
        E = -(entropy_per_cluster * Rk).sum()
        return T.squeeze(E)

    def calculate_hard_entropy(self, S):
        """
        Calculates hard entropy using numpy
        :param S the histogram vectors
        :return the hard entropy
        """

        Nc = self.mapping.get_value().shape[1]
        C = np.float64(self.C.get_value())

        nn = NearestNeighbors(n_neighbors=1).fit(C)
        distances, idx = nn.kneighbors(S)
        idx = np.squeeze(idx)
        H = np.zeros((Nc, C.shape[0]))
        for i in range(len(idx)):
            H[self.labels[i], idx[i]] += 1
        Nk = H.sum(axis=0)
        P = H / Nk

        entropy_per_cluster = P * np.log2(P)
        entropy_per_cluster[np.isnan(entropy_per_cluster)]=0

        entropy_per_cluster = entropy_per_cluster.sum(axis=0)

        Rk = Nk / Nk.sum()
        E = -(entropy_per_cluster * Rk).sum()
        return E

    def calculate_soft_entropy_debug(self, S):
        """
        Calculates soft entropy using numpy (for debug purposes)
        :param S the histogram vectors
        :return the hard entropy
        """
        C = self.C.get_value(borrow=True)
        mapping = self.mapping.get_value(borrow=True)
        Q = None
        for i in range(S.shape[0]):
            soft_similarity = np.exp(-lin.norm(S[i, :]-C, 2, axis=1)/self.m)
            soft_similarity = soft_similarity / lin.norm(soft_similarity, 1)

            if Q is None:
                Q = soft_similarity
            else:
                Q = np.vstack((Q, soft_similarity))

        Nk = np.sum(Q, axis=0)

        H = np.dot(mapping.T, Q)
        P = H / Nk

        entropy_per_cluster = P*np.log2(P)
        entropy_per_cluster[np.isnan(entropy_per_cluster)] = 0
        entropy_per_cluster = entropy_per_cluster.sum(axis=0)

        Rk = Nk / Nk.sum()
        E = -(entropy_per_cluster * Rk).sum()

        return E


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
    aa = T.sum(A*A, axis=1)
    bb = T.sum(B*B, axis=1)
    AB = T.dot(A, T.transpose(B))

    AA = T.transpose(T.tile(aa, (bb.shape[0], 1)))
    BB = T.tile(bb, (aa.shape[0], 1))

    D = AA + BB - 2*AB
    # D = T.fill_diagonal(D, 0)
    D = T.maximum(D, 0)
    D = T.sqrt(D)
    return D


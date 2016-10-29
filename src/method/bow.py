
from lasagne.updates import adam
import theano
import numpy as np
from tqdm import tqdm
from scipy import io as sio

from sklearn.neighbors import NearestNeighbors

import theano.tensor as T
import theano.gradient
import time
from entropy import SoftEntropy
import sklearn.cluster as cluster
floatX = theano.config.floatX

floatX = theano.config.floatX

class SoftBoW:
    """
    Defines a Soft BoW input layer
    """

    def __init__(self, g=0.1, feature_dimension=128, n_codewords=16):

        # Setup parameters
        self.Nk = n_codewords
        self.D = feature_dimension
        self.g = g

        # Create a randomly initialized dictionary
        V = np.asanyarray(np.random.rand(self.Nk, self.D), dtype=floatX)
        self.V = theano.shared(value=V, name='V', borrow=True)

        # Tensor of input objects
        self.X = T.tensor3(name='X', dtype=floatX)

        # Feature matrix of an object
        self.x = T.matrix(name='x', dtype=floatX)

        # Encode a set of objects (the number of features per object is fixed and same for all objects)
        self.encode_objects_theano = theano.function(inputs=[self.X], outputs=self.sym_histograms(self.X))

        # Encodes only one object with an arbitrary number of features
        self.encode_object_theano = theano.function(inputs=[self.x], outputs=self.sym_histogram(self.x))

    def sym_histogram(self, X):
        """
        Computes a  a soft-quantized histogram of a set of feature vectors (matrix X)
        """
        distances = symbolic_distance_matrix(X, self.V)
        membership = T.nnet.softmax(-distances/self.g)
        histogram = T.mean(membership, axis=0)
        return histogram

    def sym_histograms(self, X):
        """
        Encodes a list of objects (matrices of feature vectors with the same number of feature vectors)
        """
        histograms, updates = theano.map(self.sym_histogram, X)
        return histograms

    def encode_objects(self, X):
        """
        Encodes a set of objects with arbitary number of features
        """
        histograms = []
        for x in X:
            histograms.append(self.encode_object_theano(np.asarray(x, dtype=floatX)))
        return np.asarray(histograms, dtype=floatX)

    def encode_objects_hard(self, X):
        """
        Implements hard BoW quantization and computes the histograms of objects in X
        :param X: features vectors the objects to be encoded
        :return: the resulting histograms
        """
        histograms = []
        V = self.V.get_value(borrow=True)
        nn = NearestNeighbors(n_neighbors=1).fit(V)

        for i in range(len(X)):
            distances, idx = nn.kneighbors(X[i])
            histogram = np.histogram(idx, np.arange(self.Nk))[0]
            histogram = histogram / np.sum(histogram, dtype=np.floatX)
            histograms.append(histogram)

        return np.asarray(histograms)

    def encode_objects_debug(self, X):
        """
        Implements soft BoW quantization and computes the histograms of objects in X
        :param X: features vectors the objects to be encoded
        :return: the resulting histograms
        """

        V = self.V.get_value(borrow=True)
        histograms = []
        for i in range(len(X)):

            D = np.zeros((X[i].shape[0], V.shape[0]))
            for j in range(V.shape[0]):
                D[:, j] = np.sqrt(np.sum((X[i] - V[j, :]) ** 2, axis=1))
            D = np.exp(-D / self.g)

            histogram = np.mean(D / np.sum(D, axis=1).reshape(-1, 1), axis=0)
            histograms.append(histogram)

        return np.asarray(histograms)

    def initialize_dictionary(self, X, init_method='k-means', max_iter=100, redo=5, n_samples=50000):
        """
        Uses the vectors in X to initialize the dictionary
        """

        if init_method not in {'k-means'}:
            raise ValueError("Codebook initialization method not supported: " + init_method)

        samples_per_object = np.ceil(n_samples / len(X))

        features = None
        print "Sampling feature vectors..."
        for i in (range(len(X))):
            idx = np.random.permutation(X[i].shape[0])[:samples_per_object]
            cur_features = X[i][idx, :]
            if features is None:
                features = cur_features
            else:
                features = np.vstack((features, cur_features))

        print "Clustering feature vectors..."
        features = np.float64(features)
        if init_method == 'k-means':
            V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo)
            self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


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



class Entropy_BoW(SoftBoW):
    """
    Defines the Entropy Optimized BoW Model
    """

    def __init__(self, g=0.1, m=0.01, feature_dimension=128, n_codewords=16, n_feature_samples=100, eta=0.01, labels=[]):

        SoftBoW.__init__(self, g=g, feature_dimension=feature_dimension, n_codewords=n_codewords)

        self.entropy = SoftEntropy(m=m, labels=labels)
        self.entropy_loss = None
        self.learning_rate = eta
        self.n_feature_samples = n_feature_samples

        # Histograms
        self.S = self.sym_histograms(self.X)

        # Entropy loss
        self.entropy_loss = self.entropy.sym_entropy(self.S)

        # Compile loss function
        self.calculate_loss_theano = theano.function([self.X], self.entropy_loss)

        # Define gradients w.r.t. V (and take care of NaNs)
        entropy_grad = T.grad(self.entropy_loss, self.S)
        entropy_grad = T.switch(T.isnan(entropy_grad), 0, entropy_grad)
        dictionary_grad = T.grad(self.entropy.sym_entropy(self.S), self.V, known_grads={self.S: entropy_grad})
        dictionary_grad = T.switch(T.isnan(dictionary_grad), 0, dictionary_grad)

        # Define and compile the training function
        self.updates = adam([dictionary_grad], [self.V], learning_rate=self.learning_rate)
        self.train_theano = theano.function(inputs=[self.X], outputs=[self.entropy_loss], updates=self.updates)


    def initialize(self, data, n_samples=30000):
        """
        Initializes the EntropyBoW object by selecting the initial centers for the dictionary and the entropy centers
        """
        data = subsample(data, self.n_feature_samples)
        self.initialize_dictionary(data, n_samples=n_samples)
        self.entropy.init_centers(self.encode_objects_theano(data))

    def train(self, data, iters=100):
        """
        Train the Soft BoW layer using the entropy objective
        """
        for iter in tqdm(range(iters)):

            subsampled_data = subsample(data, self.n_feature_samples)
            print subsampled_data.shape
            cur_loss = self.train_theano(subsampled_data)[0]
            print "Loss at iteration ", iter, " = ", cur_loss



    def calculate_entropy(self, data):
        """
        Calculates the entropy of the data
        """
        S = self.encode_objects(data)
        soft_entropy = self.entropy.calculate_soft_entropy(S)
        soft_entropy = soft_entropy.tolist()
        hard_entropy = self.entropy.calculate_hard_entropy(S)
        return soft_entropy, hard_entropy


def subsample(X, n_samples=100):
    """
    Subsamples the objects using n_samples from each object
    """
    objects = []

    for x in X:
        features = x

        # If less than n_samples features exist, repeat some of them
        if x.shape[0] < n_samples:
            n_repeats = np.int64(np.ceil(n_samples/x.shape[0]))
            for i in range(n_repeats):
                features = np.vstack((features, x))

        # Get a sample of the permutation of the features
        idx = np.random.permutation(features.shape[0])
        idx = idx [:n_samples]
        objects.append(features[idx, :])
    return np.asarray(objects, dtype=floatX)


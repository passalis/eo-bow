from lasagne.updates import adam
import theano
import numpy as np
from tqdm import tqdm
import theano.tensor as T
import theano.gradient
from entropy import SoftEntropy
import sklearn.cluster as cluster
from utils import symbolic_distance_matrix, subsample

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
        self.encode_objects_theano = theano.function(inputs=[self.X], outputs=self._sym_histograms(self.X))

        # Encodes only one object with an arbitrary number of features
        self.encode_object_theano = theano.function(inputs=[self.x], outputs=self._sym_histogram(self.x))

    def _sym_histogram(self, X):
        """
        Computes a  a soft-quantized histogram of a set of feature vectors (matrix X)
        """
        distances = symbolic_distance_matrix(X, self.V)
        membership = T.nnet.softmax(-distances / self.g)
        histogram = T.mean(membership, axis=0)
        return histogram

    def _sym_histograms(self, X):
        """
        Encodes a list of objects (matrices of feature vectors with the same number of feature vectors)
        """
        histograms, updates = theano.map(self._sym_histogram, X)
        return histograms

    def transform(self, X):
        """
        Encodes a set of objects with arbitary number of features
        """
        histograms = []
        for x in X:
            histograms.append(self.encode_object_theano(np.asarray(x, dtype=floatX)))
        return np.asarray(histograms, dtype=floatX)

    def initialize_dictionary(self, X, max_iter=100, redo=5, n_samples=50000):
        """
        Uses the vectors in X to initialize the dictionary
        """

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
        V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo)
        self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


class Entropy_BoW(SoftBoW):
    """
    Defines the Entropy Optimized BoW Model
    """

    def __init__(self, labels, g=0.1, m=0.01, feature_dimension=128, n_codewords=16, n_feature_samples=100, eta=0.01):
        """
        The labels of the objects used for the optimization.
        The objects must be in the same order when the fit function is called
        :param labels: labels of the objects used for the optimization
        :param g: BoW quantization parameter
        :param m: entropy softness parameter
        :param feature_dimension: dimension of the extracted feature vectors
        :param n_codewords: number of codewords in the dictionary
        :param n_feature_samples: number of feature vectors to use in each iteration
        :param eta: learning rate
        """

        SoftBoW.__init__(self, g=g, feature_dimension=feature_dimension, n_codewords=n_codewords)

        self.entropy = SoftEntropy(m=m, labels=labels)
        self.entropy_loss = None
        self.learning_rate = eta
        self.n_feature_samples = n_feature_samples

        # Histograms
        self.S = self._sym_histograms(self.X)

        # Entropy loss
        self.entropy_loss = self.entropy._sym_entropy(self.S)

        # Compile loss function
        self.calculate_loss_theano = theano.function([self.X], self.entropy_loss)

        # Define gradients w.r.t. V (and take care of NaNs)
        entropy_grad = T.grad(self.entropy_loss, self.S)
        entropy_grad = T.switch(T.isnan(entropy_grad), 0, entropy_grad)
        dictionary_grad = T.grad(self.entropy._sym_entropy(self.S), self.V, known_grads={self.S: entropy_grad})
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

    def fit(self, data, iters=100):
        """
        Train the Soft BoW layer using the entropy objective
        """
        for iter in tqdm(range(iters)):
            subsampled_data = subsample(data, self.n_feature_samples)
            cur_loss = self.train_theano(subsampled_data)[0]
            print "Loss at iteration ", iter, " = ", cur_loss

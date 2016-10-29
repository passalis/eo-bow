from sklearn.neighbors import NearestNeighbors
import numpy as np
import cPickle
from sklearn.metrics.pairwise import additive_chi2_kernel

class Database(object):

    def __init__(self, database_vectors, targets):
        """
        Creates a database object that contains the database_vectors and their labels
        :param database_vectors: the objects of the database
        :param targets: the labels of the database vectors (used to determine the relevant queries)
        """

        self.nn = NearestNeighbors(n_neighbors=database_vectors.shape[0], algorithm='brute', metric='euclidean')
        self.nn.fit(database_vectors)
        self.targets = np.cast[np.int](targets)
        bins = np.bincount(self.targets)
        idx = np.nonzero(bins)[0]
        self.instances_per_target = dict(zip(idx, bins[idx]))
        self.number_of_instances = float(len(targets))
        self.recall_levels = np.arange(0, 1.01, 0.1)
        self.fine_recall_levels = np.arange(0, 1.01, 0.05)

    def _get_binary_relevances(self, queries, targets):
        distances, indices = self.nn.kneighbors(queries)
        relevant_vectors = np.zeros_like(indices)
        for i in range(targets.shape[0]):
            relevant_vectors[i, :] = self.targets[indices[i, :]] == targets[i]
        return relevant_vectors

    def _get_metrics(self, relevant_vectors, targets):
        # Calculate precisions per query
        precision = np.cumsum(relevant_vectors, axis=1) / np.arange(1, self.number_of_instances + 1)

        # Calculate recall per query
        instances_per_query = np.zeros((targets.shape[0], 1))
        for i in range(targets.shape[0]):
            instances_per_query[i] = self.instances_per_target[targets[i]]
        recall = np.cumsum(relevant_vectors, axis=1) / instances_per_query

        # Calculate interpolated precision
        interpolated_precision = np.zeros_like(precision)
        for i in range(precision.shape[1]):
            interpolated_precision[:, i] = np.max(precision[:, i:], axis=1)

        # Calculate precision @ 11 recall point
        precision_at_recall_levels = np.zeros((targets.shape[0], self.recall_levels.shape[0]))
        for i in range(len(self.recall_levels)):
            idx = np.argmin(np.abs(recall - self.recall_levels[i]), axis=1)
            precision_at_recall_levels[:, i] = interpolated_precision[np.arange(targets.shape[0]), idx]

        # Calculate the means values of the metrics
        ap = np.mean(precision_at_recall_levels, axis=1)

        # Return the mAP
        return np.mean(ap)

    def evaluate(self, queries, targets):
        """
        Evaluates the performance of the representation using the given queries and the returns the mAP
        :param queries: the query vectors
        :param targets: the labels of the queries (used to determine the relevant vectors)
        :return: the mAP
        """
        relevant_vectors = self._get_binary_relevances(queries, targets)
        map = self._get_metrics(relevant_vectors, targets)
        return map

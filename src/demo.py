import numpy as np
from utils.datasets import load_features_15scene, extract_features_15scene, get_15scene_splits
from method.bow import Entropy_BoW
from utils.database_evaluation import Database

def run_bow(seed=1):
    np.random.seed(1)

    features, labels = load_features_15scene()
    features = np.asarray(features)

    labels = np.int32(labels)
    train_idx, test_idx = get_15scene_splits(seed=seed)

    print "Data loaded!"
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    test_features = features[test_idx]
    test_labels = labels[test_idx]

    bow = Entropy_BoW(n_codewords=64, labels=train_labels, feature_dimension=128)

    print "Learning initial dictionary..."
    bow.initialize(train_features)
    print "Encoding objects..."
    Strain = bow.encode_objects(train_features)
    Stest = bow.encode_objects(test_features)

    print "Evaluating representation..."
    database = Database(Strain, train_labels)
    print "mAP = ", database.evaluate(Stest, test_labels)

    bow.train(train_features, iters=30)

    print "Encoding objects..."
    Strain = bow.encode_objects(train_features)
    Stest = bow.encode_objects(test_features)

    print "Evaluating optimized representation..."
    database = Database(Strain, train_labels)
    print "mAP = ", database.evaluate(Stest, test_labels)

if __name__ == '__main__':
    extract_features_15scene()
    run_bow()
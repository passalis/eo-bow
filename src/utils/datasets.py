from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from tqdm import tqdm
from utils.dsift import DsiftExtractor
from scipy import misc
import cPickle as pickle

def extract_features_15scene():
    """
    Extracts dense SIFT features from the 15-scene dataset and stores it as a pickled file
    :return:
    """
    #16
    sift_extractor = DsiftExtractor(8, 16)

    files, labels, categories = load_15_scene_dataset(path='/home/nick/local/datasets/15_scene/scene_categories')
    descriptors = []
    print "Extracting features..."
    for i in tqdm(range(len(files))):
        im = np.float64(misc.imread(files[i]))
        feaArr, positions = sift_extractor.process_image(im)
        descriptors.append(np.float32(feaArr))

    with open("dataset.pickle", "wb") as f:
        pickle.dump(descriptors, f)
        pickle.dump(labels, f)

    print len(files), len(labels)
    print labels
    print categories

def load_features_15scene():
    with open("dataset.pickle", "rb") as f:
        descriptors = pickle.load(f)
        labels = pickle.load(f)
    print len(descriptors)
    print  descriptors[0].shape
    return descriptors, labels


def load_15_scene_dataset(path='/home/nick/local/datasets/15_scene/scene_categories'):
    """
    Loads the 15scene dataset from the path
    :param path:
    :return:
    """
    categories = [f for f in listdir(path) if isdir(join(path, f))]
    files = []
    labels = []

    for i, cat in enumerate(categories):
        c_files = [join(join(path, cat), f) for f in listdir(join(path, cat))]
        files.extend(c_files)
        labels.extend([ i for x in c_files])

    labels = np.asarray(labels)
    return files, labels, categories


def get_15scene_splits(n_train=100, seed=1, path='/home/nick/local/datasets/15_scene/scene_categories'):
    """
    Generates random splits for evaluating the 15-scene dataset
    The evaluation procedure must be repeated 5 times
    :param n_train: number of train data per class (default 100)
    :param seed: use a didferent seed to get different splits
    :param path: path of 15scene dataset for loading the data
    :return:
    """
    files, labels, cats = load_15_scene_dataset(path)
    np.random.seed(seed)
    labels = np.asarray(labels)

    # Get splits
    train_idx = []
    test_idx = []

    for i in range(15):
        idx = (labels == i)
        idx = np.where(idx)[0]
        np.random.shuffle(idx)
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.asarray(train_idx)
    np.random.shuffle(train_idx)
    test_idx = np.asarray(test_idx)
    np.random.shuffle(test_idx)

    return train_idx, test_idx
import dircache
import os
import numpy as np
import sklearn as sk
import re
from sklearn.decomposition import PCA
from spro_utils import load_mfcc_file

__author__ = 'blazej'

#load the data from given input dir and transform the files into single datapoints with concatenated frames
def getlabels_for_filenames(filenames, labels):
    labels = dict(labels)
#    filenames with the original extension and no path
    just_filenames = [os.path.basename(f)[:-5] + "aiff" for f in filenames]
    y = np.array([int(labels[f]) for f in just_filenames])
    return y


def readlabels(filenames, labels_path):
    labels = np.loadtxt(open(labels_path, "r"), dtype='str', delimiter=",", skiprows=1)
    y = getlabels_for_filenames(filenames, labels)
    return y

def calculateAuc(classifier, x, y):
    predictions = classifier.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = sk.metrics.roc_curve(y, predictions, pos_label=1)
    return sk.metrics.auc(fpr,tpr)


def load_data(inputdirpath, labels_path):
    filenames = list_dir(inputdirpath, None)
    y = readlabels(filenames, labels_path)
    return np.vstack([load_mfcc_file(f).data.flatten() for f in filenames]), y, filenames

def load_x(inputdirpath):
    filenames = list_dir(inputdirpath, None)
    filenames.sort(numeric_cmp)
    return np.vstack([load_mfcc_file(f).data.flatten() for f in filenames])


def list_dir(inputdirpath, maxchars = None):
    file_list = dircache.listdir(inputdirpath)
    print 'found', len(file_list), 'files in ', inputdirpath
    return [os.path.join(inputdirpath, filename) for filename in file_list if filename.endswith('fbank')][:maxchars]


def runpca(train_x):
    pca = PCA(n_components=train_x.shape[1], copy=False, whiten=True)
    new_x = pca.fit_transform(train_x)
    return new_x, pca


if __name__ == '__main__':
    pass
    



def numeric_cmp(x,y):
    x = re.search(r'[0-9]+', x).group(0)
    y = re.search(r'[0-9]+', y).group(0)
    return int(x).__cmp__(int(y))

    
    



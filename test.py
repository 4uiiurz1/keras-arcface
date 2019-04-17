import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler

import archs
from metrics import *


def main():
    # dataset
    (X, y), (X_test, y_test) = mnist.load_data()

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
    y_ohe = keras.utils.to_categorical(y, 10)
    y_ohe_test = keras.utils.to_categorical(y_test, 10)

    # feature extraction
    softmax_model = load_model('models/mnist_vgg8_3d/model.hdf5')
    softmax_model = Model(inputs=softmax_model.input, outputs=softmax_model.layers[-2].output)
    softmax_features = softmax_model.predict(X_test, verbose=1)
    softmax_features /= np.linalg.norm(softmax_features, axis=1, keepdims=True)

    cosface_model = load_model('models/mnist_vgg8_cosface_3d/model.hdf5', custom_objects={'CosFace': CosFace})
    cosface_model = Model(inputs=cosface_model.input[0], outputs=cosface_model.layers[-3].output)
    cosface_features = cosface_model.predict(X_test, verbose=1)
    cosface_features /= np.linalg.norm(cosface_features, axis=1, keepdims=True)

    arcface_model = load_model('models/mnist_vgg8_arcface_3d/model.hdf5', custom_objects={'ArcFace': ArcFace})
    arcface_model = Model(inputs=arcface_model.input[0], outputs=arcface_model.layers[-3].output)
    arcface_features = arcface_model.predict(X_test, verbose=1)
    arcface_features /= np.linalg.norm(arcface_features, axis=1, keepdims=True)

    sphereface_model = load_model('models/mnist_vgg8_sphereface_3d/model.hdf5', custom_objects={'SphereFace': SphereFace})
    sphereface_model = Model(inputs=sphereface_model.input[0], outputs=sphereface_model.layers[-3].output)
    sphereface_features = sphereface_model.predict(X_test, verbose=1)
    sphereface_features /= np.linalg.norm(sphereface_features, axis=1, keepdims=True)

    # plot
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    for c in range(len(np.unique(y_test))):
        ax1.plot(softmax_features[y_test==c, 0], softmax_features[y_test==c, 1], softmax_features[y_test==c, 2], '.', alpha=0.1)
    plt.title('Softmax')

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    for c in range(len(np.unique(y_test))):
        ax2.plot(arcface_features[y_test==c, 0], arcface_features[y_test==c, 1], arcface_features[y_test==c, 2], '.', alpha=0.1)
    plt.title('ArcFace')

    fig3 = plt.figure()
    ax3 = Axes3D(fig3)
    for c in range(len(np.unique(y_test))):
        ax3.plot(cosface_features[y_test==c, 0], cosface_features[y_test==c, 1], cosface_features[y_test==c, 2], '.', alpha=0.1)
    plt.title('CosFace')

    fig4 = plt.figure()
    ax4 = Axes3D(fig4)
    for c in range(len(np.unique(y_test))):
        ax4.plot(sphereface_features[y_test==c, 0], sphereface_features[y_test==c, 1], sphereface_features[y_test==c, 2], '.', alpha=0.1)
    plt.title('SphereFace')

    plt.show()


if __name__ == '__main__':
    main()

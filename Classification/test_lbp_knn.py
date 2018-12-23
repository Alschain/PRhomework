#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from skimage import feature as ft
from skimage import color,transform
from skimage.feature import local_binary_pattern
from skimage.io import imread,imsave,imshow

model_folder = 'model'
features_name = 'features.pkl'
labels_name = 'labels.pkl'

radius = 1
n_points = 8 * radius

def knn(features, labels, predicted_sample, k):
    '''
    knn algorithm
    '''
    N = len(features)
    repeated_vector = np.array([predicted_sample]).repeat(N, axis=0)
    # get distance to all training data
    dist = np.linalg.norm(features-repeated_vector, ord=2, axis=1)
    # sort distance array
    sorted_dist = dist.argsort()
    vote_dict = {}
    # for the shortest k sample, vote the most frequent label
    for i in range(k):
        if labels[sorted_dist[i]] in vote_dict.keys():
            vote_dict[labels[sorted_dist[i]]] += 1
        else:
            vote_dict[labels[sorted_dist[i]]] = 1
    vote_class_index = max(vote_dict, key=vote_dict.get)
    return vote_class_index

def lbpknnpredict(filename):
    with open(os.path.join(model_folder,features_name),'rb') as feafile:
        features = pickle.load(feafile)
    with open(os.path.join(model_folder,labels_name),'rb') as lbfile:
        labels = pickle.load(lbfile)
    img = imread(filename)
    img = transform.resize(img, (32, 32))
    gray = color.rgb2gray(img)
    lbp = local_binary_pattern(gray, n_points, radius, 'default')
    imsave('lbp.jpg',lbp/255.0)
    max_bins = int(lbp.max() + 1)
    test_feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    predicted_label = knn(features, labels, test_feature, 16)
    return predicted_label
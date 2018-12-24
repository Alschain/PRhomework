#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from skimage import feature as ft
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.io import imread
from tqdm import tqdm

data_folder = 'data'
model_folder = 'model'
train_batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
test_batches = ['test_batch']

radius = 1
n_points = 8 * radius

def unpickle(file):
    '''
    load dataset
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_and_labels(train_batches):
    '''
    get data and labels for given files
    '''
    data = []
    labels = []
    for train_batch in train_batches:
        filename = os.path.join(data_folder,train_batch)
        batch_dict = unpickle(filename)
        # pixel of each image
        batch_feature = batch_dict[b'data']
        # label of each image
        batch_label = batch_dict[b'labels']
        batch_feature_num = len(batch_feature)
        batch_label_num = len(batch_label)
        # batch_feature = batch_feature.reshape(-1,32,32,3)
        for i in range(batch_feature_num):
            data.append(batch_feature[i])
        for i in range(batch_label_num):
            labels.append(batch_label[i])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def extract_features(data):
    features = []
    N = len(data)
    for i in tqdm(range(N)):
        # reshape data to get rgb array
        rgb = data[i].reshape(3,-1).T.reshape(-1,32,32,3)
        gray = color.rgb2gray(rgb)[0]
        # get lbp features
        lbp = local_binary_pattern(gray, n_points, radius, 'default')
        max_bins = int(lbp.max() + 1)
        # get lbp histogram as features
        feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        features.append(feature)
    return np.array(features)

def knn_predict(features, labels, predicted_sample, k):
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

data, labels = get_data_and_labels(train_batches)
features = extract_features(data)
test_data, test_labels = get_data_and_labels(test_batches)
test_features = extract_features(test_data)

with open(os.path.join(model_folder, 'features.pkl'),'wb+') as feafile:
    pickle.dump(features, feafile)
with open(os.path.join(model_folder, 'labels.pkl'),'wb+') as lbfile:
    pickle.dump(labels, lbfile)

test_num = len(test_features)
predict_labels = []
K = 16
for i in tqdm(range(test_num)):
    predict_labels.append(knn_predict(features, labels, test_features[i], K))
result = [int(predict_labels[i]==test_labels[i]) for i in range(test_num)]
acc = sum(result)/len(result)
print(acc)

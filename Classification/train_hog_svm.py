#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from skimage import feature as ft
from skimage import color
from skimage.io import imread
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

data_folder = 'cifar-10-batches-py'
model_folder = 'model'
train_batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
test_batches = ['test_batch']
model_name = 'hog_svm.model'

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
        batch_feature = batch_dict[b'data']
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
        # get hog features
        feature = ft.hog(gray, feature_vector=True)
        features.append(feature)
    return np.array(features)

data, labels = get_data_and_labels(train_batches)
features = extract_features(data)
test_data, test_labels = get_data_and_labels(test_batches)
test_features = extract_features(test_data)

svc_rbf = SVC(kernel='rbf', C=1e2, gamma=0.1)
# train svm model
model = OneVsRestClassifier(svc_rbf,-1).fit(features, labels)
# save model
with open(os.path.join(model_folder,model_name),'wb+') as mdfile:
    pickle.dump(model, mdfile)
acc = model.score(test_features,test_labels)
print(acc)
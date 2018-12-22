#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from skimage import feature as ft
from skimage import color, transform
from skimage.io import imread, imsave

model_folder = 'model'
model_name = 'hog_svm.model'

def predict(filename):
    img = imread(filename)
    img = transform.resize(img, (32, 32))
    gray = color.rgb2gray(img)
    f, image = ft.hog(gray, visualise=True)
    feature = np.array(ft.hog(gray, feature_vector=True).reshape(-1,324))
    imsave('./hog.jpg', image)
    with open(os.path.join(model_folder,model_name), 'rb') as mdfile:
        model = pickle.load(mdfile)
    predicted_label = model.predict(feature)
    return predicted_label[0]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:46:41 2020

@author: adam

Unpickles all files and combines into one shuffled array.

Builds one .dat file for each video file consisting of dictionary with keys:
    1. data
    2. labels
"""

import glob
import pickle
import numpy as np
import os


files = glob.glob("/data_preprocessed_python/*.dat")

eegs = []
labels = []

for each_path in files:
    dat = pickle.load(open(each_path, 'rb'), encoding='iso-8859-1')
    eegs.append(dat['data'])
    labels.append(dat['labels'])

eegs_combined = np.asarray(eegs).reshape((32 * 40, 40, 8064))
labels_combined = np.asarray(labels).reshape((32 * 40, 4))

random_indexes = np.random.permutation(32 * 40)
eegs_combined = eegs_combined[random_indexes]
labels_combined = labels_combined[random_indexes]

if not os.path.isdir('./modular'):
    os.makedirs('./modular')

for idx, pair in enumerate(zip(eegs_combined, labels_combined)):
    data = {
        'data': pair[0],
        'label': pair[1]
    }
    
    with open('./modular/' + str(idx) + '.dat', 'wb') as f:
        pickle.dump(data, f)

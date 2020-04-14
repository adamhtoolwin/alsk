#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:46:41 2020

@author: adam
Unpickles all files and combines into one shuffled array.

Builds dataset into two files:
    1. eeg.dat for eeg sequences
    2. labels.dat for labels
"""

import glob
import pickle
import numpy as np


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

with open('eegs.dat', 'wb') as f:
    pickle.dump(eegs_combined, f)
    
with open('labels.dat', 'wb') as f:
    pickle.dump(labels_combined, f)
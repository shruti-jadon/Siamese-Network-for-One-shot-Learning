#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:05:56 2017

@author: shruti
"""

import pickle
from sklearn.metrics import silhouette_samples, silhouette_score

fc = pickle.load( open( "RELU_100.pickle", "rb" ) )

for key in fc.keys():
    print (key)
    print (fc[key])
    
s_avg = silhouette_score(fc["numpy_all"], fc["numpy_labels"])
print (s_avg)

# RELU 0.7766 #0.7573
# KAF 0.8052   10: 0.75 
# KAF2D 0.81641 10: 0.723

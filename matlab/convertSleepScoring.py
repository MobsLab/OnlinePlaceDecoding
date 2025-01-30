import os
import sys
import numpy as np
from scipy.io import savemat

path = sys.argv[1]

savemat(path + '/sleepScoring.mat',
    {'channels': np.load(path+'/channels.npy'),
    'sleepState': np.load(path+'/text.npy'),
    'values': np.load(path+'/metadata.npy'),
    'timestamps':np.load(path+'/metadata.npy')})
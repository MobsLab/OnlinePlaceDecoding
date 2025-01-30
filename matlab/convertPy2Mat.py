import os
import sys
import numpy as np
from scipy.io import savemat

constructFolderName = lambda path, s: s if path[-1]=='/' else constructFolderName(path[:-1], path[-1]+s)
findFolderName = lambda path: constructFolderName(path, '')

path = sys.argv[1]
if path[-1]=='/':
    path = path[:-1]
folderName = findFolderName(path)


data = {}
for r,d,f in os.walk(path):
    for file in f:
        if '.npy' in file:
            data[file[:-4]] = np.load(path + '/' + file)

savemat(path + '/'+ folderName +'.mat', data)
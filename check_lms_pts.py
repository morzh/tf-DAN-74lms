
from os import listdir
from os.path import isfile, join
import os.path
import os

import numpy as np 

path_faceset = '/media/morzh/ext4_volume/data/Faces/data/images_all_74lms/'


files = [f for f in listdir(path_faceset) if isfile(join(path_faceset, f))]

for file in files:

    ext = file[-3:]

    if ext == 'pts':

        lms = np.genfromtxt(path_faceset+file, skip_header=3, skip_footer=1)

        if lms.shape[0] != 74:

            print file



'''
path_faceset = '/media/morzh/ext4_volume/data/Faces/data/tf_DAN_74lms/'


files = [f for f in listdir(path_faceset) if isfile(join(path_faceset, f))]

for file in files:

    ext = file[-3:]

    if ext == 'ptv':

        lms = np.loadtxt(path_faceset+file)

        print lms.shape

        if lms.shape[0] != 74:

            print file


'''
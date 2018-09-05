
import numpy as np
import cv2
import utils
from os import listdir
from os.path import isfile, join
import os.path
import os
from matplotlib import pyplot as plt



path_faceset_in = '/media/morzh/ext4_volume/data/Faces/all_in_one/set_004/'
path_faceset_out= '/media/morzh/ext4_volume/data/Faces/all_in_one/set_004_rect/'


files = [f for f in listdir(path_faceset_in) if isfile(join(path_faceset_in, f))]


# points_pts = pts_loader.load(file_pts)
# print np.array(points_pts)

for file in  files:

    ext = file[-3:]

    if ext == 'jpg' or ext == 'png' or ext == 'JPG': 

        if not os.path.isfile(path_faceset_in+file+'.rect'):

            print path_faceset_in+file, ' --> skipping'
            continue

        rect = np.loadtxt(path_faceset_in+file+'.rect')
        img = cv2.imread(path_faceset_in+file)

        rect = (np.round(rect)).astype(np.int32)
        img = img[ rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        '''
        plt.imshow(img)
        plt.show()
        '''
        cv2.imwrite(path_faceset_out+file, img)




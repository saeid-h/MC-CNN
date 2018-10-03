#!/usr/bin/python

import os
import cv2 as cv
import numpy as np
import pfmutil as pfm
from scipy import misc
import cpputils

def get_filenames(dir_path):
    f = []
    for (_, _, filenames) in os.walk(dir_path):
        f.extend(filenames)
    return f

working_dir = "/home/saeid/00 Datasets/KITTI2012/training/"
left_path = working_dir + "image_0/"
right_path = working_dir + "image_1/"
gt_path = working_dir + "disp_occ/"
disp_path = working_dir + "disp_mccnn-kitti-fast/"


max_disp = 228


f = get_filenames(gt_path)

ext = f[0].split(".")[-1]
fnames = list()
for i in range(len(f)):
    parsedName = f[i].split(".")
    if len(parsedName) > 2:
        print ("Parsing left image name Error: There is a dot in filename.")
    if parsedName[-1] != ext:
        print ("Parsing left imagee Error: There are more than single type available")
    fnames.append(parsedName[0])
fnames.sort()

#os.system ("echo \"\nTemporary output file for Freiburg disparity construction.\n\" > tempout.txt")

model = "./main.lua kitti fast -a predict " + \
        "-net_fname " + "net/net_kitti_fast_-a_train_all.t7 " 


for i in range (len(fnames)):
    left_img = left_path + fnames[i] + "." + ext
    right_img = right_path + fnames[i] + "." + ext
    command = model + " -left \"" + left_img + "\""+ \
        " -right \"" + right_img + "\"" +\
        " -disp_max " + str(max_disp) #+ " -sm_terminate cnn"

    
    height, width = misc.imread(left_img).shape
    os.system(command)
    #os.system ("cp disp.png " + disp_path + fnames[i] + ".png")
    
    disp = np.memmap('./disp.bin', dtype=np.float32, shape=(height, width))
    cpputils.write2png(disp,disp_path + fnames[i] + ".png")
    #misc.imsave (disp_path + fnames[i] + ".png", disp.astype('uint8'))








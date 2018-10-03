#!/usr/bin/python

import os
import cv2 as cv
import numpy as np
import pfmutil as pfm


def get_filenames(dir_path):
    f = []
    for (_, _, filenames) in os.walk(dir_path):
        f.extend(filenames)
    return f

working_dir = "/home/saeid/00 Datasets/Freiburg/"
left_path = working_dir + "driving/frames_cleanpass/15mm_focallength/scene_forwards/slow/left/"
right_path = working_dir + "driving/frames_cleanpass/15mm_focallength/scene_forwards/slow/right/"
gt_path = working_dir + "driving/disparity/15mm_focallength/scene_forwards/slow/left/"
disp_path = working_dir + "driving/disparity/15mm_focallength/scene_forwards/slow/disp_mccnn-kitti-fast/"

width = 540
height = 960
max_disp = 365


f = get_filenames(left_path)
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

model = "./main.lua kitti fast -a predict -net_fname net/net_kitti_fast_-a_train_all.t7 " 



for i in range (len(fnames)):
    left_img = left_path + fnames[i] + "." + ext
    right_img = right_path + fnames[i] + "." + ext
    command = model + " -left \"" + left_img + "\""+ \
        " -right \"" + right_img + "\"" +\
        " -disp_max " + str(max_disp) #+ " >> tempout.txt"

    os.system(command)
    disp = np.memmap('./disp.bin', dtype=np.float32, shape=(width, height))
    pfm.save (disp_path + fnames[i] + ".pfm", disp)



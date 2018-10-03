

import numpy as np
from scipy.misc import imsave

left = np.memmap('./left.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
right = np.memmap('./right.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
disp = np.memmap('./disp.bin', dtype=np.float32, shape=(1, 1, 370, 1226))

left = np.argmin(left[0,:,:,:], axis=0)
right = np.argmin(right[0,:,:,:], axis=0)
disp = disp[0,0,:,:]

imsave("left.png", left)
imsave("right.png", right)
imsave("disp.png", disp)

print (left.shape, right.shape, disp.shape)

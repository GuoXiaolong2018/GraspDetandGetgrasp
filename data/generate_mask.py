import numpy as np 
import scipy.misc   
mask = np.zeros((480,640))
mask[60:400,120:600]=1
scipy.misc.imsave('data/mask.jpg', mask)
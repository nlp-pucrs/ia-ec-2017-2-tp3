# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.preprocessing import normalize


im = Image.open('Image_0_01.tif')
print(im.size)
# ainda precisa normalizar os valores lidos
imarray = np.array(im)

#patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
#    random_state=0)
#patches.shape
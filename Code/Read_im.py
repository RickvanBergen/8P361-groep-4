from PIL import Image
import Util as ut
import os
import numpy as np

path = ut.folder_path()
dir = os.listdir(path)
for i in dir:
    im = Image.open(os.path.join(path,i))
    arrim = np.array(im)
    im.show()
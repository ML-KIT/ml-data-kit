import os
import tarfile
import time
import sys
import scipy.io
import numpy as np
from PIL import Image
import re
import h5py
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_data():
    if not os.path.exists(os.path.join(os.getcwd(), "jpg")):
        if not os.path.exists(os.path.join(os.getcwd(), "102flowers.tgz")):
            print ('Downloading Flowers data from  http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz ...')
            urlretrieve ('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',  '102flowers.tgz', reporthook)
        print ('\nExtracting 102flowers.tgz ...')
        tfile = tarfile.open (os.path.join(os.getcwd(), "102flowers.tgz"), 'r:gz')
        tfile.extractall ('.')
        tfile.close()


    if not os.path.exists(os.path.join(os.getcwd(), 'setid.mat')):
        print ('Downloading the data splits from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat ...')
        urlretrieve ('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat', 'setid.mat')
    if not os.path.exists(os.path.join(os.getcwd(), 'imagelabels.mat')):
        print ('Downloading the image labels from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat ...')
        urlretrieve ('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', 'imagelabels.mat')
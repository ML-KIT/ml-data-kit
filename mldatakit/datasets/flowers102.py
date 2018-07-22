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

def resize(npy_array, image_shape):
    result = []
    for img in npy_array:
        im = Image.fromarray(img)
        im = im.resize(image_shape, Image.ANTIALIAS)
        result.append(np.asarray(im))
    return result

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

def load_images(folder, splitid, imagelabels, train_test_val):
    print ('Loading {} images ... '.format(train_test_val), end='', flush=True)
    images = []
    image_names = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            img = Image.open(subdir + '/' + filename)
            file_index = int(re.split(r'[._]+', filename)[1])
            if int(file_index) in splitid:
                image_names.append(np.string_(filename))
                np_img = np.array(img)
                images.append(np_img)
                labels.append(imagelabels[file_index-1])
                img.close()
                
    images = np.array(images)
    images = resize(images, (256,256))
    image_names = np.array(image_names)
    labels = np.array(labels)
    print ('Done')
    return images, image_names, labels

def h5_creator (filename, images, image_names, labels ):
    print ('Creating {} ... '.format(filename), end='', flush=True)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('x', compression="gzip", data=images)
        hf.create_dataset('y', compression="gzip", data=labels)
        hf.create_dataset('image_names', compression="gzip", data=image_names)
    hf.close()
    print ('Done')

def get_data():
    download_data()
    setid = scipy.io.loadmat('setid.mat')
    trnid = setid['trnid'][0]
    valid = setid['valid'][0]
    tstid = setid['tstid'][0]
    imagelabels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    train_images, train_image_names, train_labels = load_images('jpg', trnid, imagelabels, 'Training')
    val_images, val_image_names, val_labels = load_images('jpg', valid, imagelabels, 'Validation')
    test_images, test_image_names, test_labels = load_images('jpg', tstid, imagelabels, 'Testing')

    h5_creator ('val.h5', val_images, val_image_names, val_labels)
    h5_creator ('train.h5', train_images, train_image_names, train_labels)
    h5_creator ('test.h5', test_images, test_image_names, test_labels)

if __name__ == '__main__':
    get_data()
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
    """Taken from https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    A simple reporthook() function for urllib.urlretrieve()‘s reporthook argument that shows a progressbar
    while downloading the data
    """
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
    """
    # Arguments
        npy_array: name of the h5 file 
        image_shape: A numpy array of images
    # Returns
        A list of resized numpy array of images
    """
    result = []
    for img in npy_array:
        im = Image.fromarray(img)
        im = im.resize(image_shape, Image.ANTIALIAS)
        result.append(np.asarray(im))
    return result

def download_data():
    """Downloads and Extracts flowers102 Dataset
    """
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
    """loads the images from a given folder
    # Arguments
        folder: directory where the images are stored
        splitid: list of ids for the dataset
        imagelabels: labels corresponding to each image
        train_test_val: one of "train", "test", "val"
    # Returns
        images: A numpy array of the images
        image_names: A numpy array of the image names
        labels: A numpy array of the labels
    """
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

def h5_creator (filename, x, y, image_names=np.array([])):
    """Creates a H5 file and datasets with all the arguments.
    # Arguments
        filename: Name of the h5 file 
        images: A numpy array of the images
        image_names: A numpy array of the image names
        labels: A numpy array of the labels
    """
    print ('Creating {} ... '.format(filename), end='', flush=True)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('x', compression="gzip", data=x)
        hf.create_dataset('y', compression="gzip", data=y)
        hf.create_dataset('image_names', compression="gzip", data=image_names)
    hf.close()
    print ('Done')

def load_data(expanded=False):
    """Downloads the data loads all the images and the labels
    # Returns
        Tuple of Numpy arrays
        if expanded is true: (x_train, y_train, train_image_names),
                (x_val, y_val, val_image_names), (x_test, y_test, test_image_names)
        if expanded is false: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    # Arguments
        expanded: Boolean, where to load expanded entities
    """ 
    download_data()
    setid = scipy.io.loadmat('setid.mat')
    trnid = setid['trnid'][0]
    valid = setid['valid'][0]
    tstid = setid['tstid'][0]
    imagelabels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

    x_train, train_image_names, y_train = load_images('jpg', trnid, imagelabels, 'Training')
    x_val, val_image_names, y_val = load_images('jpg', valid, imagelabels, 'Validation')
    x_test, test_image_names, y_test = load_images('jpg', tstid, imagelabels, 'Testing')

    if expanded == False:
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    else:
        return (x_train, y_train, train_image_names), (x_val, y_val, val_image_names), \
                (x_test, y_test, test_image_names)

def create_h5(expanded=False):
    if expanded == False:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(expanded=False)
        h5_creator ('train.h5', x=x_train, y=y_train)
        h5_creator ('val.h5', x=x_val, y=y_val)
        h5_creator ('test.h5', x=x_test, y=y_test)

    else:
        (x_train, y_train, train_image_names), (x_val, y_val, val_image_names), \
                (x_test, y_test, test_image_names) = load_data(expanded=True)
        h5_creator ('train.h5', x=x_train, y=y_train, image_names=train_image_names)
        h5_creator ('val.h5', x=x_val, y=y_val, image_names=val_image_names)
        h5_creator ('test.h5', x=x_test, y=y_test, image_names=test_image_names)


if __name__ == '__main__':
    create_h5()
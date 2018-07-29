import os
import sys
from PIL import Image
import glob
import numpy as np
import h5py
import csv
import time
import zipfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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

def download_data():
    """Downloads and Extracts tiny-imagenet Dataset
    """
    if not os.path.exists(os.path.join(os.getcwd(), "tiny-imagenet-200")):
        if not os.path.exists(os.path.join(os.getcwd(), "tiny-imagenet-200.zip")):
            print ('Downloading Flowers data from  http://cs231n.stanford.edu/tiny-imagenet-200.zip ...')
            urlretrieve ('http://cs231n.stanford.edu/tiny-imagenet-200.zip',  'tiny-imagenet-200.zip', reporthook)
        print ('\nExtracting tiny-imagenet-200.zip ...', end='', flush=True)
        zfile = zipfile.ZipFile (os.path.join(os.getcwd(), 'tiny-imagenet-200.zip'), 'r')
        zfile.extractall ('.')
        zfile.close()
        print ('Done')


def get_word_labels():
    """Get the wnids and label names from the words.txt file.
    # Returns
        A dictionary where keys are the wnids and values are the label names
    """
    file = open ('tiny-imagenet-200/words.txt', 'r')
    word_labels = {}
    for f in file:
        f = f.split('	')
        words = f[1]
        words = words.replace('\n', '')
        word_labels[f[0]] = words
    file.close()
    return word_labels

def get_train_wnid():
    """Extracts the wnids from the subdirectories for every image in the train folder
    # Returns
        A dictionary where keys are the image names and values are the wnids
    """
    wnid_labels = {}
    for subdir, dirs, files in os.walk('tiny-imagenet-200/train'):
        for filename in files:
            if filename.endswith(('.txt')):
                file = open(subdir + '/' +filename, 'r')
                for line in file:
                    line = line.split('	')
                    wnid_labels[line[0]] = subdir.split('/')[-1]
                file.close()
    return wnid_labels

def get_val_wnid():
    """Extracts the wnids from the val_annotations.txt file for every image in the val folder
    # Returns
        A dictionary where keys are the image names and values are the wnids
    """
    file = open('tiny-imagenet-200/val/val_annotations.txt', 'r')
    wnid_labels = {}
    for f in file:
        f = f.split('	')
        wnid_labels[f[0]] = f[1]
    file.close()
    return wnid_labels

def load_labels():
    """Gets wnids for every image and convert them to categorical
    # Returns
        train_wnid: A dictionary where keys are the training image names and values are the wnids
        val_wnid: A dictionary where keys are the validation image names and values are the wnids
        uniq_wnids: A list of all the wnids
    """
    train_wnid = get_train_wnid()
    val_wnid = get_val_wnid() 
    uniq_wnids = list(set(list(train_wnid.values()) + list(val_wnid.values())))
    return train_wnid, val_wnid, uniq_wnids

def load_images (folder, wnid_labels, uniq_wnids):
    """loads the images from a given folder
    # Arguments
        folder: directory where the images are stored
        wnid_labels: A dictionary where keys are the validation image names and values are the wnids
        uniq_wnids: A list of all the wnids
    # Returns
        images: A numpy array of the images
        image_names: A numpy array of the image names
        labels: A numpy array of the labels
        wnids: A numpy array of the wnids
        label_names: A numpy array of the label names
    """
    word_labels = get_word_labels()
    images = []
    labels = []
    wnids = []
    label_names = []
    image_names = []
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.JPEG', '.jpeg', '.JPG', '.jpg', '.PNG', '.png')):
                img = Image.open(subdir + '/' + filename)
                np_img = np.array(img)
                if np_img.ndim == 2:
                    np_img = np.dstack([np_img]*3)
                images.append(np_img)
                filename = filename.split("/")[-1]
                labels.append(uniq_wnids.index(wnid_labels[filename]))
                image_names.append(np.string_(filename))
                wnids.append(np.string_(wnid_labels [filename]))
                label_names.append(np.string_(word_labels [wnid_labels[filename]]))
                img.close()
                if (len(images)%5000) is 0: print ('{} imges processed'.format(len(images)))
    images = np.array(images)
    labels = np.array(labels)
    wnids = np.array(wnids)
    image_names = np.array(image_names)
    label_names = np.array(label_names)
    print ('Image processing finished')
    return images, image_names, labels, wnids, label_names

def h5_creator (filename, images, image_names, labels, wnids, label_names ):
    """Creates a H5 file and datasets with all the arguments.
    # Arguments
        filename: name of the h5 file 
        images: A numpy array of the images
        image_names: A numpy array of the image names
        labels: A numpy array of the labels
        wnids: A numpy array of the wnids
        label_names: A numpy array of the label names
    """
    print ('Creating {} ... '.format(filename), end='', flush=True)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('x', compression="gzip", data=images)
        hf.create_dataset('y', compression="gzip", data=labels)
        hf.create_dataset('image_names', compression="gzip", data=image_names)
        hf.create_dataset('label_names', compression="gzip", data=label_names)
        hf.create_dataset('wnids', compression="gzip", data=wnids)
    hf.close()

def load_data(expanded=False):
    """Downloads the data loads all the images and the labels
    # Returns
        Tuple of Numpy arrays
        if expanded is true: (x_train, y_train), (x_val, y_val)
        if expanded is false: (x_train, y_train, train_image_names, train_wnids, train_label_names),
                (x_val, y_val, val_image_names, val_wnids, val_label_names)
    """ 
    download_data()
    train_wnid_labels, val_wnid_labels, uniq_wnids = load_labels()

    x_val, val_image_names, y_val, val_wnids, val_label_names = load_images ('tiny-imagenet-200/val', val_wnid_labels, uniq_wnids)
    x_train, train_image_names, y_train, train_wnids, train_label_names = load_images ('tiny-imagenet-200/train', train_wnid_labels, uniq_wnids)
    if expanded == False:
        return (x_train, y_train), (x_val, y_val)
    else:
        return (x_train, y_train, train_image_names, train_wnids, train_label_names), \
            (x_val, y_val, val_image_names, val_wnids, val_label_names)

if __name__ == '__main__':
    (x_train, y_train, train_image_names, train_wnids, train_label_names), \
            (x_val, y_val, val_image_names, val_wnids, val_label_names) = load_data(expanded=True)  
    h5_creator ('val.h5', val_images, val_image_names, val_labels, val_wnids, val_label_names)
    h5_creator ('train.h5', train_images, train_image_names, train_labels, train_wnids, train_label_names)

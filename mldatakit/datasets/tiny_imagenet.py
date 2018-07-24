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
    file = open ('tiny-imagenet-200/words.txt', 'r')
    word_labels = {}
    for f in file:
        f = f.split('	')
        words = f[1]
        words = words.replace('\n', '')
        word_labels[f[0]] = words
    file.close()
    return word_labels

def get_train_wnid_labes():
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

def get_val_wnid_labes():
    file = open('tiny-imagenet-200/val/val_annotations.txt', 'r')
    wnid_labels = {}
    for f in file:
        f = f.split('	')
        wnid_labels[f[0]] = f[1]
    file.close()
    return wnid_labels

def load_labels():
    train_wnid_labels = get_train_wnid_labes()
    val_wnid_labels = get_val_wnid_labes() 
    categorical_labels = list(set(list(train_wnid_labels.values()) + list(val_wnid_labels.values())))
    return train_wnid_labels, val_wnid_labels, categorical_labels

def load_images (folder, wnid_labels, categorical_labels):
    word_labels = get_word_labels()
    # if train_val == 'train': wnid_labels = get_train_wnid_labes()
    # elif train_val == 'val': wnid_labels = get_val_wnid_labes()
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
                labels.append(categorical_labels.index(wnid_labels[filename]))
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
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('x', compression="gzip", data=images)
        hf.create_dataset('y', compression="gzip", data=labels)
        hf.create_dataset('image_names', compression="gzip", data=image_names)
        hf.create_dataset('label_names', compression="gzip", data=label_names)
        hf.create_dataset('wnids', compression="gzip", data=wnids)
    hf.close()

def get_data(): 
    download_data()
    train_wnid_labels, val_wnid_labels, categorical_labels = load_labels()

    val_images, val_image_names, val_labels, val_wnids, val_label_names = load_images ('tiny-imagenet-200/val', val_wnid_labels, categorical_labels)
    train_images, train_image_names, train_labels, train_wnids, train_label_names = load_images ('tiny-imagenet-200/train', train_wnid_labels, categorical_labels)
    
    h5_creator ('val.h5', val_images, val_image_names, val_labels, val_wnids, val_label_names)
    h5_creator ('train.h5', train_images, train_image_names, train_labels, train_wnids, train_label_names)

if __name__ == '__main__':
    get_data()
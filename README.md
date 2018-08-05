# ML_Data_Kit

*mldatakit* is a python package focussed on interconvertibility between data formats and access to public datasets, requiring minimal effort from Machine Learning researchers and Developers.

## Getting public datasets

The package also lets you download public datasets with minimal effort and convert it to h5 format. Currently you can download and convert two public image datasets from the official source.

Import the package in the following way to download, unzip and convert the data to h5.

### Tiny Imagenet  
  
Dataset of 100,000  64x64 color training images, labeled over 200 categories, 10,000 validation images.

##### To load the datasets

```
from mldatakit.datasets import tiny_imagenet

(x_train, y_train), (x_val, y_val) = tiny_imagenet.load_data(expanded=False)  

# or, 

(x_train, y_train, train_image_names, train_wnids, train_label_names), \
        (x_val, y_val, val_image_names, val_wnids, val_label_names) = tiny_imagenet.load_data(expanded=True)  
```
* Returns (Tuple of Numpy arrays)
    * if expanded is False: (x_train, y_train), (x_val, y_val)
    * if expanded is True: (x_train, y_train, train_image_names, train_wnids, train_label_names),
            (x_val, y_val, val_image_names, val_wnids, val_label_names)

* Arguments
    * expanded: Boolean, where to load expanded entities

##### To create the h5 files

```
tiny_imagenet.create_h5(expanded=True)
```
* Creates `train.h5` and `val.h5` with the following entities

    * x: A numpy array of the images
    * y: A numpy array of the labels
    * image_names (if expanded is True): A numpy array of the image names
    * wnids (if expanded is True): A numpy array of the wnids
    * label_names (if expanded is True): A numpy array of the label names
* Arguments
    * expanded: Boolean, where to load expanded entities

### Flowers102 

Dataset of 1,020  256x256 color training images, labeled over 102 categories, 1,020 validation images and 6,149 test images.

##### To load the datasets

```
from mldatakit.datasets import flowers102

(x_train, y_train), (x_val, y_val), (x_test, y_test) = flowers102.load_data(expanded=False)

# or,

(x_train, y_train, train_image_names), (x_val, y_val, val_image_names), \
        (x_test, y_test, test_image_names) = flowers102.load_data(expanded=True)
```

* Returns (Tuple of Numpy arrays)

    * if expanded is false: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    * if expanded is true: (x_train, y_train, train_image_names),
            (x_val, y_val, val_image_names), (x_test, y_test, test_image_names)
* Arguments
    * expanded: Boolean, where to load expanded entities

##### To create the h5 files

```
flowers102.create_h5(expanded=True)
```
* Creates `train.h5`, `val.h5` and `test.h5` with the following entities

    * x: A numpy array of the images
    * y: A numpy array of the labels
    * image_names (if expanded is True): A numpy array of the image names
* Arguments
    * expanded: Boolean, where to load expanded entities
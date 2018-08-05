# ML_Data_Kit

*mldatakit* is a python package focussed on interconvertibility between data formats and access to public datasets, requiring minimal effort from Machine Learning researchers and Developers.

## Getting public datasets

The package also lets you download public datasets with minimal effort and convert it to h5 format. Currently you can download and convert two public image datasets from the official source.

Execute the following command in terminal or import the package in the following way to download, unzip and convert the data to h5.

#### Tiny Imagenet  
  

```
from mldatakit.datasets import tiny_imagenet

(x_train, y_train, train_image_names, train_wnids, train_label_names), \
        (x_val, y_val, val_image_names, val_wnids, val_label_names) = tiny_imagenet.load_data(expanded=True)  
# or,

(x_train, y_train), (x_val, y_val) = tiny_imagenet.load_data(expanded=False)  
```
* Returns (Tuple of Numpy arrays)
    * if expanded is True: (x_train, y_train, train_image_names, train_wnids, train_label_names),
            (x_val, y_val, val_image_names, val_wnids, val_label_names)
    * if expanded is False: (x_train, y_train), (x_val, y_val)

* Arguments
    * expanded: Boolean, where to load expanded entities

to create the h5 files

```
tiny_imagenet.h5_creator ('train.h5', x=x_train, y=y_train, image_names=train_image_names,\
		wnids=train_wnids, label_names=train_label_names)
tiny_imagenet.h5_creator ('val.h5', x=x_val, y=y_val, image_names=val_image_names, \
		wnids=val_wnids, label_names=val_label_names)
```
* Arguments
    * filename: name of the h5 file 
    * x: A numpy array of the images
    * y: A numpy array of the labels
    * image_names: A numpy array of the image names
    * wnids: A numpy array of the wnids
    * label_names: A numpy array of the label names

#### Flowers102 


  
```
from mldatakit.datasets import flowers102

(x_train, y_train, train_image_names), (x_val, y_val, val_image_names), \
        (x_test, y_test, test_image_names) = flowers102.load_data(expanded=True)
# or,

(x_train, y_train), (x_val, y_val), (x_test, y_test) = flowers102.load_data(expanded=False)
```

* Returns (Tuple of Numpy arrays)
    * if expanded is true: (x_train, y_train, train_image_names),
            (x_val, y_val, val_image_names), (x_test, y_test, test_image_names)
    * if expanded is false: (x_train, y_train), (x_val, y_val), (x_test, y_test)
* Arguments
    * expanded: Boolean, where to load expanded entities

to create the h5 files

```
flowers102.h5_creator ('train.h5', x=x_train, y=y_train, image_names=train_image_names)
flowers102.h5_creator ('val.h5', x=x_val, y=y_val, image_names=val_image_names)
flowers102.h5_creator ('test.h5', x=x_test, y=y_test, image_names=test_image_names)
```
* Arguments
    * filename: Name of the h5 file 
    * x: A numpy array of the images
    * y: A numpy array of the labels
    * image_names (Optional): A numpy array of the image names

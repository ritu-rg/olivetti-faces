# -*- coding: utf-8 -*-
"""
Last amended: 21/12/2017
Myfolder: C:\Users\ashok\OneDrive\Documents\ridge_regression

Objectives:
    Reconstruct half-faces using regression techniques
    Learn to use multi-output regressors
    
Data file:
    Data file is fetched from Internet automatically
    
What next?
    Try the general class of multi-output regressor:
        http://scikit-learn.org/dev/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor
        And attempt the problem here:
            https://www.kaggle.com/c/afsis-soil-properties/data    
"""


## 0. Clear memory
%reset -f
import gc
gc.collect()

# 1.0 Call libraries
# For data manipulation
import numpy as np

# 1.1 For plotting faces
import matplotlib.pyplot as plt   
from skimage.io import imshow

# 1.2 Our dataset is here
from sklearn.datasets import fetch_olivetti_faces

# 1.3 Regressors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor


# 2.0 Load the faces datasets
# There are ten different images of each of 40 distinct subjects.
#  That is each subject has 10 images but taken differently..
#  For some subjects, the images were taken at different times,
#   varying the lighting, facial expressions (open / closed eyes,
#    smiling / not smiling) and facial details (glasses / no glasses).
#     All the images were taken against a dark homogeneous background
#     with the subjects in an upright, frontal position (with tolerance 
#      for some side movement).
data = fetch_olivetti_faces()  # downloads the data archive from AT&T

# 2.1 This is a sepcial data class
type(data)                     # sklearn.utils.Bunch
# Bunch: Dictionary-like object, the interesting attributes are: ‘data’,
#       the data to learn, ‘target’, the classification labels,
#      ‘target_names’, the meaning of the labels, ‘feature_names’,
#       the meaning of the features, and ‘DESCR’, the full description
#       of the dataset. In R dictionary is like a list in R.

# 3 Extract data components
# 3.1 Target first
targets = data.target          # Data target
type(targets)
targets.size                   # 400 images

# 4. Images next
data.images                    # Images set
data.images.shape              # Image is 400X 64 X 64

# 4.1 See an image
firstImage = data.images[0]
imshow(firstImage) 


# 5.0 Flatten each image
data = data.images.reshape(data.images.shape[0], data.images.shape[1] * data.images.shape[2])     # 64 X 64 = 4096
# 5.1 Flattened 64 X 64 array
data.shape                                # 400 X 4096

# 6.0 Patition datasets into two (fancy indexing)
targets < 30                # Output is true/false
train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300
test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100


# 7.0 Test on a subset of people
#     Generate 10 random integers between 0 and 100
n_faces = test.shape[0]//10             # // is unconditionally "flooring division",
n_faces
face_ids = np.random.randint(0 , 100, size =n_faces)
face_ids
# 7.1 So we have n_faces random-faces from within 1 to 100
test = test[face_ids, :]   
test

# 8.0 Total pixels in any image
n_pixels = data.shape[1]
n_pixels

# 8.1 Select upper half of the faces as predictors
X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",
X_train
                                            #    3.1//1.2 = 2.0
# 8.2 Lower half of the faces will be target(s)                 
y_train = train[:, n_pixels // 2:]
y_train

# 9.0 Similarly for test data. Upper and lower half
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

# 9. Fit multi-output estimators
#  Experiment with more mulit-output regressors (such as RandomForestRegressor)
#  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

# Prepare a dictionary of estimators after instantiating each one of them
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,     # Out of 20000
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "RandomForest": RandomForestRegressor(),
}


# 9.1 Create an empty dictionary to collect prediction values
y_test_predict = dict()

# 10. Fit each model by turn and make predictions
#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)                    # fit() with instantiated object
    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name
                                                       # Note that output of estimator.predict(X_test) is prediction for
                                                       #  all the test images and NOT one (or one-by-one)
# 10.1 Just check    
y_test_predict

# 10.2 Just check shape of one of them
y_test_predict['Ridge'].shape    # 5 X 2048    
y_test_predict['K-nn'].shape
y_test_predict['Extra trees'].shape
y_test_predict['Linear regression'].shape
y_test_predict['RandomForest']. shape


## Processing output
# 11. Each face should have this dimension
image_shape = (64, 64)

## 11.1 For 'Ridge' regression
##      We will have total images as follows:
#      Per esimator, we will have n_faces * 2
#      So total - n_estimators * n_faces * 2
#      Fig size should be accordingly drawn

# 11.2 Total faces per estimator: 2 * n_faces

plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
    

## 12. For 'Extra trees' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()

## 13. For 'Linear regression' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()



## For '"K-nn' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()


## For '"Random Forest' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['RandomForest'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()



## Preprocessing step 1 - filtered images and xml files
import os
import time
start_pascal = time.time()

img_names = []
xml_names = []

for dirname, subdirs, filenames in os.walk('asset/dataset/CPY/'):
  for filename in filenames:
    if filename[-3:] != "xml":
      img_names.append(filename)
    else:
      xml_names.append(filename)

print("Total Files")
print("Images    :",len(img_names))
print("Xml Files :",len(xml_names))
print()

## Preprocessing step 2 - cropped images by bounding box using xml files 
import xmltodict
from matplotlib import pyplot as plt
from skimage.io import imread

path_annotations = "asset/dataset/CPY/"
path_images = "asset/dataset/CPY/"
# 'bengal','persian','siamese','ragdoll','rblue','sphynx'
class_names = ['bengal','ragdoll','siamese','rblue']
images = []
target = []
gmb = []
cim= []

# 'bengal','siamese','ragdoll' 0.78


def crop_bounding_box(img, bnd):
  x1, x2, y1, y2 = list(map(int, bnd.values()))
  _img = img.copy()
  _img = _img[y1:y2, x1:x2]
  _img = _img[:,:,:3]
  return _img

for img_name in img_names:
  with open(path_annotations+img_name[:-4]+".xml") as fd:
    doc = xmltodict.parse(fd.read())

  img = imread(path_images+img_name)
  temp = doc["annotation"]["object"]
  if type(temp) == list:
    for i in range(len(temp)):
      if temp[i]["name"] not in class_names:
        continue
      images.append(crop_bounding_box(img, temp[i]["bndbox"]))
      cim.append(crop_bounding_box(img, temp[i]["bndbox"]))
      gmb.append(img)
      target.append(temp[i]["name"])
  else:
    if temp["name"] not in class_names:
        continue
    images.append(crop_bounding_box(img, temp["bndbox"]))
    cim.append(crop_bounding_box(img, temp["bndbox"]))
    gmb.append(img)
    target.append(temp["name"])

# print total target by class
print("Total target by class")
for i in class_names:
  print(i , "   :", target.count(i))

# --------------------------------------------------------------------
end_pascal = time.time()
pascalex = end_pascal - start_pascal
print("Execution time Read Pascal : {} seconds".format(pascalex) ,"\n" )

# --------------------------------------------------------------------

start_prepo = time.time()
## Preprocessing step 3 - resize images to 258x258 and normalize (remove background and grayscale)
import numpy as np
import skimage
from skimage.transform import resize

from scipy import ndimage
from skimage import io, color, exposure, filters
from skimage.filters import unsharp_mask

def resize_image(img, size = 32):
  _img = img.copy() 
  _img = resize(_img, (size, size))
  return _img

def remove_background(img):
  _img = img.copy()
  thresh = skimage.filters.threshold_otsu(_img)
  _img = (_img > thresh).astype(np.float32)
  return _img

def grayscale(img):
  _img = img.copy()
  _img = np.dot(_img[...,:3], [0.299, 0.587, 0.114])
  return _img

def prepo(img):
  _img = img.copy()

  # Apply Adaptive Histogram Equalization (AHE) to the grayscale image
  clahe_image = exposure.equalize_adapthist(_img , clip_limit=3)

  # Apply Unsharp Masking to the AHE result
  blurred = ndimage.gaussian_filter(clahe_image, sigma=1)
  unsharp_maskk = clahe_image - 0.5 * blurred

  # Perform segmentation using Thresholding
  threshold_value = unsharp_mask(unsharp_maskk, radius=5, amount=2)
  binary_image = unsharp_maskk > threshold_value

  # Convert the binary image to uint8 and scale it to 0-255
  binary_image = binary_image.astype(np.float32)
  return _img  


for i in range(len(images)):
  images[i] = resize_image(images[i])
  images[i] = grayscale(images[i])
  images[i] = prepo(images[i])
  # images[i] = remove_background(images[i])

## Extration step 1 - extract features using PHOG (Pyramid Histogram of Oriented Gradients)
from PIL import Image
from scipy.ndimage import convolve

def phog(img, bin_size=16, levels=3):
    # Compute the gradient magnitude and orientation
    gx = convolve(img, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = convolve(img, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi

    # Binning
    # ---------------------------------------------------------------------------
    # The gradient orientation is divided into bin_size bins using integer division
    binned_orientation = (orientation / bin_size).astype(np.int32) % bin_size

    # Pyramidal representation
    # ---------------------------------------------------------------------------
    # downsampling the image using the pyrDown function and computing the histograms of
    # oriented gradients for each level. The histograms are stored in a list pyramid.
    pyramid = []
    def pyr_down(img, bin_size=8):
        # Define the downsampling kernel

        # The values in the 5x5 array are chosen based on the Gaussian function, which is a symmetric bell-
        # shaped curve that has a peak at the center and falls off symmetrically in both directions.
        kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])

        # Normalize the kernel based on the factor
        kernel = 1.0/bin_size * kernel

        # Convolve the image with the kernel

        #  mode = 'constant' means that the values of the image at the edges
        #  are assumed to be a constant value, which is typically set to 0.
        convolved = convolve(img, kernel, mode='constant')

        # Downsample the image by taking every other row and column
        downsampled = convolved[::2, ::2]

        return downsampled


    for i in range(levels):
        histograms = np.zeros((bin_size,))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                histograms[binned_orientation[y, x]] += magnitude[y, x]
        pyramid.append(histograms)
        img = pyr_down(img)

    # Step 5: Normalization
    # ---------------------------------------------------------------------------

    normalized_pyramid = []
    for histograms in pyramid:
        normalization_factor = np.sum(histograms**2)**0.5
        if normalization_factor > 1e-12:
            histograms /= normalization_factor
        normalized_pyramid.append(histograms)

    # Step 6: Concatenation
    # ---------------------------------------------------------------------------

    phog_descriptor = np.concatenate(normalized_pyramid)

    # Step 7: Representation (linear vector)
    # ---------------------------------------------------------------------------

    return phog_descriptor

#--------------------------------------------------------------------------------
import cv2
def calculate_lbp(img, radius=3, neighbors=8):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    lbp = np.zeros_like(img)
    
    for i in range(radius, img.shape[0] - radius):
        for j in range(radius, img.shape[1] - radius):
            center = img[i, j]
            binary_code = 0
            
            for k in range(neighbors):
                x = i + int(radius * np.cos(2 * np.pi * k / neighbors))
                y = j - int(radius * np.sin(2 * np.pi * k / neighbors))
                
                if img[x, y] >= center:
                    binary_code |= (1 << (neighbors - 1 - k))
            
            lbp[i, j] = binary_code
    
    return lbp


#--------------------------------------------------------------------------------
# features = []
# for i in range(len(images)):
#   print("Processing", i+1, "of",len(images))
#   features.append(calculate_lbp(images[i]))


features = []
for i in range(len(images)):
    print("Processing", i+1, "of", len(images))
    lbp = calculate_lbp(images[i])  # Menghitung LBP untuk gambar
    
    # Mengubah dimensi LBP menjadi 1D
    lbp_1d = lbp.reshape(-1)
    
    features.append(lbp_1d)

# Mengubah dimensi fitur menjadi 2D
features_2d = np.array(features)


# --------------------------------------------------------------------
end_prepo = time.time()
prepotime = end_prepo - start_prepo
print()
print("Execution time Prepp : {} seconds".format(prepotime))
# --------------------------------------------------------------------

start_split = time.time()
from sklearn.model_selection import train_test_split

X, y = features_2d, np.array(target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print()
print("Training Data :\n", np.asarray(np.unique(y_train, return_counts=True)).T)
print("Test Data     :\n", np.asarray(np.unique(y_test, return_counts=True)).T)

end_split = time.time()
splittime = end_split - start_split
print()
print("Execution Time Split Data : {} seconds".format(splittime))

# Classification step 1
# ------------------------------------------------------
# !             TEST TUNING CLASSIFIER
# ------------------------------------------------------
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV

# clf = StackingClassifier(
#     estimators=[('svm', SVC(random_state=42)),
#                 ('tree', DecisionTreeClassifier(random_state=42))],
#     final_estimator=LogisticRegression(random_state=42),
#     n_jobs=-1)

# param_grid = {
#     'svm__C': [1.6, 1.7, 1.8],
#     'svm__kernel': ['rbf','linear'],
#     'tree__criterion': ['entropy','gini'],
#     'tree__max_depth': [9, 10, 11],
#     'final_estimator__C': [1.3, 1.4, 1.5]
# }

# grid = GridSearchCV(
#     estimator=clf,
#     param_grid=param_grid,
#     scoring='accuracy',
#     n_jobs=-1)

# grid.fit(X_train, y_train)

# print('Best parameters: %s' % grid.best_params_)
# print('Accuracy       : %.2f' % grid.best_score_)



# ------------------------------------------------------

# # # ------------------------------------------------------
# # #!             RUN CLASSIFIER
# # # ------------------------------------------------------
start_class = time.time()
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# clf = LinearSVC(multi_class='crammer_singer', dual=False)
clf = SVC(C=1, kernel='rbf',random_state=42, max_iter=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print()
print('Accuracy score   : ', accuracy_score(y_test, y_pred))
print('Precision score  : ', precision_score(y_test, y_pred, average='weighted'))
print('Recall score     : ', recall_score(y_test, y_pred, average='weighted'))
print('F1 score         : ', f1_score(y_test, y_pred, average='weighted'))
end_class = time.time()
classtime = end_class - start_class
print()
print("Execution time Classifier : {} seconds".format(classtime))

# ------------------------------------------------------
#!               SAVE CLASSIFIER MODEL
# ------------------------------------------------------
import pickle
pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'wb') as file:
  pickle.dump(clf, file)


# ------------------------------------------------------
#?             CONFUSSION MATRIX
# ------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
# print("Predicting cat breed on the test set")
print()
print(classification_report( y_test, y_pred, target_names=class_names ))

import pickle

pkl_filename = 'svm_model.pkl'
with open(pkl_filename, 'rb') as file:
    loaded_model = pickle.load(file)

from skimage.io import imshow,show
import random
rnd = random.randint(1, len(features))
print()
print('RAND DATA :',rnd )
# imshow(images[rnd])
# show()

pdt = features[rnd]
prediction = loaded_model.predict([pdt])
print('Len of Features : ', len(features))
print("Data Test  :", target[rnd])
print("Prediction :", prediction)

# Create a figure and axes
fig, axes = plt.subplots(1, 4, figsize=(10, 5))

# Display the first image
axes[1].imshow(images[rnd])
axes[1].set_title('Cropped ')

# Display the second image
axes[2].imshow(cim[rnd])
axes[2].set_title('Cropped Original')

axes[3].imshow(gmb[rnd])
axes[3].set_title('Original')

cmd = ConfusionMatrixDisplay.from_estimator( clf, X_test, y_test, display_labels=class_names)
cmd.plot(ax=axes[0],xticks_rotation="vertical" )

# Adjust the layout

# Show the plot
plt.tight_layout()
plt.show()

# Total target by class
# bengal      : 395 + 6   V
# ragdoll     : 324 + 82  V
# siamese     : 255 + 165 V
# rblue       : 389 + 11

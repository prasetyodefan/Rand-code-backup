## Pyramid of Histograms of Oriented Gradients (PHOG)
The PHOG descriptor is a feature descriptor used in computer vision and image processing.The function takes an input image, img, as a parameter, along with two optional parameters: bin_size and levels. bin_size determines the number of bins into which the gradient orientation is divided and defaults to 16. levels specifies the number of levels in the pyramidal representation of the image and defaults to 3. The function first converts the input image to grayscale and computes the gradient magnitude and orientation of the image using the Sobel operator. It then bins the gradient orientation into bin_size bins and computes the histograms of oriented gradients for each level of the pyramidal representation of the image. The histograms are then normalized and concatenated into a single feature vector, which is returned by the function.

The code performs the following steps:
1. Pre-processing: The input image is converted to grayscale
2. Gradient computation: The gradient magnitude and orientation are computed using the Sobel operator
3. Binning: The gradient orientation is divided into bin_size bins using integer division and modulo operations.
4. Pyramidal representation: A pyramidal representation of the image is created by successively downsampling the image using the pyrDown function and computing the histograms of oriented gradients for each level. The histograms are stored in a list pyramid.
5. Normalization: The histograms in pyramid are normalized to make the feature descriptor invariant to changes in lighting

<details><summary> Show Code </summary>
  
```python

from PIL import Image
import numpy as np
from scipy.ndimage import convolve

def phog(img, bin_size=16, levels=3):

    # Step 1: Pre-processing
    #---------------------------------------------------------------------------

    # Convert RGB to G by using the dot product of the input 
    # image with a weighting array [0.2989, 0.5870, 0.1140].
    # array represents the scaling factors for the RGB channels of the image
    def grayscale(img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
    # Convert img to grayscale with float 32 bit DataType
    gray = grayscale(img).astype(np.float32)

    # Perform a 2D convolution of an image with a kernel.
    # Parameter Usage | image
    #                 | kernel convolution kernel represent 2D np.array 
    #                 | mode 'same' output will have the same size as the input, 
    #                   with the result padded with zeros if necessary
    def convolve2d(img, kernel, mode='same'):
        m, n = img.shape
        k, l = kernel.shape
        if mode == 'same':
            pad_size = (k - 1) // 2
            pad = np.zeros((m + 2 * pad_size, n + 2 * pad_size))
            pad[pad_size:-pad_size, pad_size:-pad_size] = img
            result = np.zeros_like(img)
        else:
            pad = img
            result = np.zeros((m - k + 1, n - l + 1))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = (pad[i:i+k, j:j+l] * kernel).sum()
        return result
    
    

    # Step 2: Gradient computation
    #---------------------------------------------------------------------------
    # The gradient magnitude and orientation are computed using the Sobel operator

    # highlight areas of the image with sharp intensity changes (edges).
    def sobelx(img):
        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        return convolve2d(gray, sobel_x_kernel, mode='same')

    def sobely(img):
        sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
        return convolve2d(gray, sobel_y_kernel, mode='same')


    sobel_x = sobelx(img)
    sobel_y = sobely(img)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    gradient_orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # Step 3: Binning
    #---------------------------------------------------------------------------

    # The gradient orientation is divided into bin_size bins using integer division
    binned_orientation = (gradient_orientation /
                          bin_size).astype(np.int32) % bin_size

    # Step 4: Pyramidal representation
    #---------------------------------------------------------------------------
    # downsampling the image using the pyrDown function and computing the histograms of 
    # oriented gradients for each level. The histograms are stored in a list pyramid.

    pyramid = []
    def pyr_down(img, bin_size=16):
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
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                histograms[binned_orientation[y, x]] += gradient_magnitude[y, x]
        pyramid.append(histograms)
        gray = pyr_down(gray)

    # Step 5: Normalization
    #---------------------------------------------------------------------------

    normalized_pyramid = []
    for histograms in pyramid:
        normalization_factor = np.sum(histograms**2)**0.5
        if normalization_factor > 1e-12:
            histograms /= normalization_factor
        normalized_pyramid.append(histograms)

    # Step 6: Concatenation
    #---------------------------------------------------------------------------

    phog_descriptor = np.concatenate(normalized_pyramid)

    # Step 7: Representation (linear vector)
    #---------------------------------------------------------------------------

    return phog_descriptor


img = Image.open('cropfaces.jpg')
img = np.array(img)
#img = arrim.shape

result = phog(img)

print('Res PHOG', result)

```
  
</details>



## To see the difference between two arrays output by the PHOG Descriptor
subtract one array from the other and calculate the Euclidean distance between them. The Euclidean distance measures the difference between two points in a multidimensional space and can be used to determine how similar or dissimilar two arrays are.

Here's an example of how you can calculate the Euclidean distance between two arrays in Python:

```python

import numpy as np

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([2, 3, 4, 5, 6])

difference = array1 - array2
distance = np.linalg.norm(difference)

print("Difference:", difference)
print("Euclidean distance:", distance)

```

In this example, array1 and array2 are two arrays of length 5. The difference between the arrays is calculated by subtracting array2 from array1 and stored in the difference variable. The Euclidean distance between the arrays is calculated by taking the norm (square root of the sum of squares) of the difference variable.

The Euclidean distance will be 0 if the arrays are identical, and will increase as the difference between the arrays increases. The Euclidean distance is a common metric used in computer vision and image analysis to compare feature descriptors and determine how similar or dissimilar two images are.

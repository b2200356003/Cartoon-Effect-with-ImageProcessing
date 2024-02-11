import numpy as np
import cv2
import matplotlib.pyplot as plt

Gaussian_kernel_size = 5
Gaussian_kernel_sigma1 = 3
Gaussian_kernel_sigma2 = 5

filter_sigma = 5
filter_kernel_size = 5

Quantization_K_value = 7

epsilon = 0.000003

image1 = cv2.imread('istanbul.jpg')[..., ::-1]

size_x,size_y = np.shape(image1)[0],np.shape(image1)[1]

plt.imshow(image1)
plt.show()


# you can switch blur type with deleting comment in median one and adding comment in gaussian one
blurred_image = cv2.GaussianBlur(image1, (filter_kernel_size, filter_kernel_size),filter_sigma)
#blurred_image = cv2.medianBlur(image1,filter_kernel_size)



gaussian_kernel1 = cv2.getGaussianKernel(filter_kernel_size, Gaussian_kernel_sigma1)
gaussian_kernel1 = np.outer(gaussian_kernel1, gaussian_kernel1)
gaussian_kernel1 /= np.sum(gaussian_kernel1)

gaussian_kernel2 = cv2.getGaussianKernel(filter_kernel_size, Gaussian_kernel_sigma2)
gaussian_kernel2 = np.outer(gaussian_kernel2, gaussian_kernel2)
gaussian_kernel2 /= np.sum(gaussian_kernel2)

# Creating DoG with subtracting filters
DoG = np.subtract(gaussian_kernel1 , gaussian_kernel2)


#normalizing
image1 = image1.astype(np.float32) / 255
DoG = DoG.astype(np.float32) / 255


convolved_image_array  = np.zeros((size_x,size_y,3))


for channel in range(0,3):
    convolved_image_array[:, :, channel] = cv2.filter2D( image1[:, :, channel] , -1, DoG)




for color in range(0, 3):
    convolved_image_array[:, :, color] = np.where(
        convolved_image_array[:, :, color] > epsilon, 1, 0)

thresholded_image_array = np.empty((size_x, size_y))

for i in range(0, size_x):
    for j in range(0, size_y):
        if (convolved_image_array[i][j][0] == 1 and convolved_image_array[i][j][1] == 1 and
                convolved_image_array[i][j][2] == 1):
            thresholded_image_array[i][j] = 1
        else:
            thresholded_image_array[i][j] = 0


"""
#QUANTIZATION with HSV

gaussian_array_image2_BGR = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
gaussian_array_image2_HSV = cv2.cvtColor(gaussian_array_image2_BGR, cv2.COLOR_BGR2HSV)

data =  gaussian_array_image2_HSV[:,:,2].reshape((-1,2))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
K = 4
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
gaussian_array_image2_HSV[:,:,2] = res.reshape((gaussian_array_image2_HSV[:,:,2].shape))


gaussian_array_image2_BGR = cv2.cvtColor(gaussian_array_image2_HSV, cv2.COLOR_HSV2BGR)
Quantized_image = cv2.cvtColor(gaussian_array_image2_BGR, cv2.COLOR_BGR2RGB)
"""


#QUANTIZATION with LAB

gaussian_array_image2_BGR = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
gaussian_array_image2_Lab = cv2.cvtColor(gaussian_array_image2_BGR, cv2.COLOR_BGR2Lab)

data =  gaussian_array_image2_Lab[:,:,0].reshape((-1,2))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

ret, label, center = cv2.kmeans(data, Quantization_K_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
gaussian_array_image2_Lab[:,:,0] = res.reshape((gaussian_array_image2_Lab[:,:,0].shape))


gaussian_array_image2_BGR = cv2.cvtColor(gaussian_array_image2_Lab, cv2.COLOR_Lab2BGR)
Quantized_image = cv2.cvtColor(gaussian_array_image2_BGR, cv2.COLOR_BGR2RGB)


# inverse the threshold
thresholded_inverse_image_array = np.zeros((size_x, size_y))

thresholded_inverse_image_array[:, :] = np.where(thresholded_image_array[:, :] > 0, 0, 1)

# combine inverse and quantized matrix
last_image = np.zeros((size_x, size_y, 3))

for channel in range(0, 3):
    last_image[:, :, channel] = np.multiply(thresholded_inverse_image_array,
                                            Quantized_image[:, :, channel]) / 255

plt.imshow(last_image)
plt.show()



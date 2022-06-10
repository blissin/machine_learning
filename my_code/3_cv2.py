import numpy as np 

import cv2 

from matplotlib import pyplot as plt 

# image = cv2.imread('projectpro_noise_20.jpg',1) 

# image_bw = cv2.imread('projectpro_noise_20.jpg',0) 

# noiseless_image_bw = cv2.fastNlMeansDenoising(image_bw, None, 20, 7, 21) 

# noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image,None,20,20,7,21) 

# titles = ['Original Image(colored)','Image after removing the noise (colored)', 'Original Image (grayscale)','Image after removing the noise (grayscale)']
# images = [image,noiseless_image_colored, image_bw, noiseless_image_bw]
# plt.figure(figsize=(13,5))
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.tight_layout()
# plt.show()

import cv2
 
img = cv2.imread("noise.jpg", cv2.IMREAD_COLOR)
 
'노이즈 제거'
denoised_img1 = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 5, 10) # NLmeans
denoised_img2 = cv2.GaussianBlur(img, (5, 5), 0) # Gaussian
denoised_img3 = cv2.medianBlur(img, 5) # Median
denoised_img4 = cv2.bilateralFilter(img, 5, 50, 50) # Bilateral
 
cv2.imshow("before", img)
# cv2.imshow("after(NLmeans)", denoised_img1)
# cv2.imshow("after(Gaussian)", denoised_img2)
# cv2.imshow("after(Median)", denoised_img3)
# cv2.imshow("after(Bilateral)", denoised_img4)
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()
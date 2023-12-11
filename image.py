import cv2
from PIL import Image
import pytesseract

# OPEN IMAGE
image_file = "data/page_01.jpg"

im = Image.open(image_file)
im.show()

img = cv2.imread(image_file)

# INVERT
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", inverted_image)

inv_image_file = "temp/inverted.jpg"
inv_im = Image.open(inv_image_file)
inv_im.show()


# Binarization
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = grayscale(img)
cv2.imwrite("temp/gray.jpg", gray_image)

gray_image_file = "temp/gray.jpg"
gray_im = Image.open(gray_image_file)
gray_im.show()

thresh, im_bw = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image.jpg", im_bw)

bw_image_file = "temp/bw_image.jpg"
bw_im = Image.open(bw_image_file)
bw_im.show()


# Noise removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

no_noise_image = noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg", no_noise_image)

no_noise_image_file = "temp/no_noise.jpg"
no_noise_im = Image.open(no_noise_image_file)
no_noise_im.show()

# Dilation and Erosion
def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

eroded_image = thin_font(no_noise_image)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)

eroded_image_file = "temp/eroded_image.jpg"
eroded_im = Image.open(eroded_image_file)
eroded_im.show()

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

dilated_image = thick_font(no_noise_image)
cv2.imwrite("temp/dilated_image.jpg", dilated_image)

dilated_image_file = "temp/dilated_image.jpg"
dilated_im = Image.open(dilated_image_file)
dilated_im.show()


# OCR
ocr_result = pytesseract.image_to_string(no_noise_im)
print(ocr_result)
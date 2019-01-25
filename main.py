import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

img = np.zeros((200, 600, 3), np.uint8)
b,g,r,a = 255, 255, 255, 0

fontpath = "SanamDeklen_chaya.ttf" 
font = ImageFont.truetype(fontpath, 80)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((50, 40),  "ที่วันนี้เข้ามาดูกันน้ะ", font = font, fill = (b, g, r, a))
img = np.array(img_pil)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
size = np.size(gray_image)
skel = np.zeros(gray_image.shape,np.uint8)
 
ret,img = cv2.threshold(gray_image,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while( not done):
    eroded = cv2.erode(gray_image,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(gray_image,temp)
    skel = cv2.bitwise_or(skel,temp)
    gray_image = eroded.copy()
 
    zeros = size - cv2.countNonZero(gray_image)
    if zeros==size:
        done = True

kernel = np.ones((1, 1),np.uint8)
# _kernel = np.ones((1, 1),np.uint8)
erosion = cv2.erode(skel, kernel,iterations = 1)
# dilation = cv2.dilate(erosion, _kernel,iterations = 1)
# gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, _kernel)

cv2.imshow("res", erosion)
cv2.imwrite("1.png", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
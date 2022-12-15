import cv2
import numpy as np
image = cv2.imread('./dataset/mvtec/bottle/test/bad/IMG_Im0001_M2_C00221231-02_FEdge Defect_U5_E3173_IOOSO_UmnU.d.bmp')
output = image.copy()
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Find circles
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 1500)
# If some circle is found
if circles is not None:
   # Get the (x, y, r) as integers
   circles = np.round(circles[0, :]).astype("int")
   print(circles)
   # loop over the circles
   for (x, y, r) in circles:
      cv2.circle(output, (x, y), r, (0, 255, 0), 2)
      h,w,_ = image.shape
      mask = np.zeros(((h),(w)), np.uint8)
      cv2.circle(mask, (int(),int(y)), 354, 255, 710)
img = cv2.bitwise_and(image, image, mask= mask)
kernel2 = np.ones((5, 5), np.float32)/25
img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

cv2.imwrite("crop_IMG_Im0062_M22_C00220475-10_FEdge Defect_U4_E3310_IOOSO_UmnU.d.bmp",img)


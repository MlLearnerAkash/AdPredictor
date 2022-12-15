# 30s

from PIL import Image
from PIL import ImageOps
import cv2
import numpy
import time
import os

st = time.time()
abs_path="/home/mlserver/Downloads/30s_dataset/30s_25_7_22_2/30s_25_7_22_2"
for file in os.listdir(abs_path):
    
    im = Image.open(abs_path +"/"+file)
    border = (450, 80, 350, 65) #(390, 70, 340, 45) #(450, 90, 350, 65) #left, top, right, bottom.
    t=ImageOps.crop(im, border)
    img = cv2.cvtColor(numpy.array(t), cv2.COLOR_RGB2BGR) 
    triangle1 = numpy.array([[0, 0], [0, 100], [20, 100], [20, 0]]) #numpy.array([[0, 0], [0, 190], [70, 190], [70, 0]]) # #numpy.array([[0, 0], [0, 190], [50, 190], [50, 0]])
    triangle2 =numpy.array([[800, 0], [800, 100], [435, 100],[435,0]]) #numpy.array([[600, 0], [600, 190], [510, 190],[510,0]])# #numpy.array([[300, 0], [600, 160], [520, 160],[520,0]])
    color = [255, 255, 255, 0] 
    cv2.fillConvexPoly(img, triangle1, color)
    cv2.fillConvexPoly(img, triangle2, color)
    cv2.imwrite("/home/mlserver/patchcore_filling_30s/dataset/mvtec/bottle/train/good"+"/"+"_2_good_new_white_mask"+ file, img)
et = time.time()
elapsed_time = et - st

print('Execution time:', elapsed_time, 'seconds')


'''
border = (450, 130, 350, 65) 
t=ImageOps.crop(im, border)

img = cv2.cvtColor(numpy.array(t), cv2.COLOR_RGB2BGR)
triangle1 = numpy.array([[0, 0], [0, 100], [20, 100], [20, 0]])
triangle2 = numpy.array([[800, 0], [800, 100], [435, 100],[435,0]])
'''

#30s

# from PIL import Image
# from PIL import ImageOps
# import cv2
# import numpy
# import time
# import os

# st = time.time()

# for file in os.listdir("C:\\Users\\EmageVision\\Desktop\\all_bad\\90s_bad_filling\\"):
    
#     im = Image.open("C:\\Users\\EmageVision\\Desktop\\all_bad\\90s_bad_filling\\" +"/"+file)
#     border = (390, 70, 340, 45)
#     t=ImageOps.crop(im, border)
#     img = cv2.cvtColor(numpy.array(t), cv2.COLOR_RGB2BGR) 
#     triangle1 = numpy.array([[0, 0], [0, 190], [70, 190], [70, 0]])
#     triangle2 = numpy.array([[600, 0], [600, 190], [510, 190],[510,0]])
#     color = [255, 255, 255, 0] 
#     cv2.fillConvexPoly(img, triangle1, color)
#     cv2.fillConvexPoly(img, triangle2, color)
#     cv2.imwrite("C:\\Users\\EmageVision\\Desktop\\all_bad_preprocees_data\\90s_bad_filling\\"+"/"+ file, img)
# et = time.time()
# elapsed_time = et - st

# print('Execution time:', elapsed_time, 'seconds')

# folding model final ashu send

# from PIL import Image
# from PIL import ImageOps
# import cv2
# import numpy
# import time
# import os

# st = time.time()

# for file in os.listdir("C:\\folding_nikhil_data_and_files\\FOLDING_CARTON_DATA\\org_data_without_preprocess\\train\\good\\"):
    
#     im = Image.open("C:\\folding_nikhil_data_and_files\\FOLDING_CARTON_DATA\\org_data_without_preprocess\\train\\good\\" +"/"+file)
#     border = (350, 175, 280, 185) #old one

#     t=ImageOps.crop(im, border)
#     img = cv2.cvtColor(numpy.array(t), cv2.COLOR_RGB2BGR) 
#     triangle1 = numpy.array([[0, 0], [0, 100], [65, 100], [65, 0]])
#     triangle2 = numpy.array([[800, 0], [800, 100], [570, 100],[570,0]])
#     triangle3 = numpy.array([[510, 80], [510, 300], [140, 300],[140,80]])

#     color = [255, 255, 255, 0] 
#     cv2.fillConvexPoly(img, triangle1, color)
#     cv2.fillConvexPoly(img, triangle2, color)
#     cv2.fillConvexPoly(img, triangle3, color)
#     cv2.imwrite("C:\\Users\\EmageVision\\Desktop\\org_data_without_preprocess\\train\\good\\"+"/"+ file, img)
# et = time.time()
# elapsed_time = et - st

# print('Execution time:', elapsed_time, 'seconds')
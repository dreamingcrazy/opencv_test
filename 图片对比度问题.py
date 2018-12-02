from PIL import Image
from functools import reduce
import math
import operator
import cv2 as cv
pic1 = Image.open('./fix1.jpg')
pic2 = Image.open('./fix0.jpg')

# 转直方图
h1 = pic1.histogram()
h2 = pic2.histogram()

difference = math.sqrt(reduce(operator.add,list(map(lambda a,b:(a-b)**2,h1,h2)))/len(h1))
print(difference)
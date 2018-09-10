#! /usr/bin/env python
# -*- coding:utf-8 -*-
from pylab import *
from PIL import Image, ImageDraw
from matplotlib.pylab import plt
import numpy as np
import cv2
'''
src = np.array(Image.open("C:/Users/12914/Pictures/demo.jpg").convert("L"))
dst = Image.new("RGB", (src.shape[1], src.shape[0]), "white")
dst = np.float64(dst)
# print src.shape
plt.figure()
plt.imshow(src, "gray")
plt.show()
'''
'''
cv2.circle(dst, click_xy, 0, (0, 0, 255), 3)
cv2.imshow("output", dst)
cv2.waitKey(0)
'''

# 压缩与旋转图像
src = Image.open("C:/Users/12914/Pictures/head3.jpg")
img = src.resize((src.size[0]/3, src.size[1]/3))
img1 = img.rotate(270)
img1.save("head.jpg")

'''
# 找特征值的坐标
d = {}
d["x"] = []
d["y"] = []
img = cv2.imread("result_output.png", 0)
for row in xrange(img.shape[0]):
    for col in xrange(img.shape[1]):
        if img[row, col] == 0:
            d["x"].append(row)
            d["y"].append(col)
print len(d["x"])
print len(d["y"])
print img.shape
'''

'''
# 轮廓检测
img = cv2.imread('C:/Users/12914/Pictures/demo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
'''

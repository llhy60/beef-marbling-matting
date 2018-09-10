#! /usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
src = cv2.imread("C:/Users/12914/Pictures/yanji.jpg")
mask = np.array([(0, -1, 0), (-1, 5, -1),(0, -1, 0)])
src = cv2.filter2D(src, -1, mask)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_new = src_gray.reshape((src_gray.shape[0]*src_gray.shape[1], 1))
src_new = np.float32(src_new)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, centers = cv2.kmeans(src_new, 3, None, criteria, 10, flags)
fatModel = np.where(labels == 0, np.where(src_new > 0, 255, 0), 0).astype("uint8")
beefModel = np.where(labels == 1, 255, 0).astype("uint8")
img_beef = beefModel.reshape((src.shape[0], src.shape[1]))
dst = fatModel.reshape((src.shape[0], src.shape[1]))
dst = cv2.dilate(dst, (3, 3))
img_beef = cv2.erode(img_beef, (3, 3))
dst = cv2.dilate(dst, (3, 3))
img_beef = cv2.erode(img_beef, (3, 3))
dst = cv2.erode(dst, (3, 3))
img_beef = cv2.dilate(img_beef, (3, 3))
dst = cv2.erode(dst, (3, 3))
img_beef = cv2.dilate(img_beef, (3, 3))
image_label = label(dst, connectivity=2)
d = {}
for i in xrange(image_label.max() + 1):
    d[i] = np.sum(image_label == i)
sort_label = sorted(d.iteritems(), key=lambda x: x[1], reverse=True)
fat_excess = 3000
# print sort_label[1][1]
fat_max = np.zeros((dst.shape)).astype("uint8")
img_fat = np.zeros((src_gray.shape[0], src_gray.shape[1]))
for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if image_label[row, col] == sort_label[1][0]:
            fat_max[row, col] = 255
if sort_label[1][1] > fat_excess:
    for row in xrange(dst.shape[0]):
        for col in xrange(dst.shape[1]):
            if dst[row, col] < fat_max[row, col]:
                img_fat[row, col] = 0
            elif dst[row, col] >= fat_max[row, col]:
                img_fat[row, col] = dst[row, col] - fat_max[row, col]
else:
    for row in xrange(dst.shape[0]):
        for col in xrange(dst.shape[1]):
            img_fat[row, col] = dst[row, col]

for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if img_beef[row, col] == 255:
            img_beef[row, col] = src_gray[row, col]
        if img_fat[row, col] == 255:
            img_beef[row, col] = img_fat[row, col]
# img_beef = cv2.cvtColor(img_beef, cv2.COLOR_GRAY2BGR)
# img_beef = cv2.dilate(img_beef, (3, 3))
# img_beef = cv2.erode(img_beef, (3, 3))

B = 0
# print sort_label
l = []
for i in xrange(2,len(sort_label)):
     l.append(sort_label[i][1])
for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if fat_max[row, col] == 255:
            B += 1
W = sum(l)
# print l





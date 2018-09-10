#! /usr/bin/env python
# -*- coding:utf-8 -*-

# 分割图像前景和图像背景
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import color, morphology
'''
threshold_value = 127
threshold_max = 255
def threshold_demo(threshold_value):
    # src = cv2.imread("filename")
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(gray_src, threshold_value, threshold_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = cv2.bitwise_not(dst)
    cv2.imshow("result image", dst)

if __name__ == "__main__":
    filename = "C:/Users/12914/Pictures/demo.jpg"
    src = cv2.imread(filename)
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", src)
    cv2.namedWindow("result image", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("threshold value:", "result image", threshold_value, threshold_max, threshold_demo)
    cv2.setTrackbarPos("threshold value:", "result image", threshold_max)
    threshold_demo(0)
    cv2.waitKey(0)
'''
# K-Means
filename = "C:/Users/12914/Pictures/demo.jpg"
src = cv2.imread("C:/Users/12914/Pictures/yanji.jpg", 0)
# print src.shape
src_new = src.reshape((src.shape[0]*src.shape[1], 1))
src_new = np.float32(src_new)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(src_new, 3, None, criteria, 10, flags)
# print labels.shape
# be = np.where(labels==0, np.where(src_new>0,255,0), 0).astype("uint8")
be = np.where(labels== 2, 255, 0)
dst = be.reshape((src.shape[0],src.shape[1]))
'''
# 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# dst_mor = cv2.morphologyEx(dst, cv2.MORPH_TOPHAT, kernel)
a = cv2.erode(dst, kernel)
b = cv2.erode(a, (5,5))
c = cv2.erode(b, (5,5))
d = cv2.erode(c, (5,5))
e = cv2.erode(d, (5,5))
f = cv2.erode(e, (5,5))
g = cv2.erode(f, (5,5))
j = cv2.erode(g, (5,5))
k = cv2.erode(j, (5,5))
# print e[0,0], dst[0,0]

img = np.zeros((src.shape[0],src.shape[1]))
for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if dst[row, col] < e[row, col]:
            img[row, col] = 0
        elif dst[row, col] >= e[row, col]:
            img[row, col] = dst[row, col] - e[row, col]


plt.subplot(221),plt.imshow(src,'gray'),plt.title('original')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(dst,'gray'),plt.title('K-Means')
plt.xticks([]),plt.yticks([])

# 形态学操作
plt.subplot(223), plt.imshow(j, 'gray'), plt.title('Morphology')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(k, 'gray'),plt.title('Morphology')
plt.xticks([]),plt.yticks([])

plt.show()
'''

img_label = label(dst, connectivity=2)
# img = color.label2rgb(img_label)
# print "regions number : {0}".format(img_label.max() + 1)
px = 0
fat_max = np.zeros((dst.shape)).astype("uint8")
for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if img_label[row, col] == 2:
            fat_max[row, col] = 255
d = {}
for i in xrange(img_label.max() + 1):
    d[i] = np.sum(img_label == i)

# print d

img = np.zeros((src.shape[0], src.shape[1]))
for row in xrange(dst.shape[0]):
    for col in xrange(dst.shape[1]):
        if dst[row, col] < fat_max[row, col]:
            img[row, col] = 0
        elif dst[row, col] >= fat_max[row, col]:
            img[row, col] = dst[row, col] - fat_max[row, col]


# print px
# fat_max = morphology.remove_small_objects(dst, min_size=300, connectivity=2)

plt.subplot(221),plt.imshow(src,'gray'),plt.title('original')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(dst,'gray'),plt.title('K-Means')
plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(fat_max, 'gray'), plt.title('Matting excess fat image')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img, 'gray'), plt.title('Matting valid fat image')
plt.xticks([]), plt.yticks([])
plt.show()
#plt.savefig("result_matting_fat_img.png")

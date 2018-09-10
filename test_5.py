#! /usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
src = cv2.imread("C:/Users/12914/Pictures/yanji.jpg")
mask = np.array([(0, -1, 0), (-1, 5, -1), (0, -1, 0)])
src = cv2.filter2D(src, -1, mask)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_new = src_gray.reshape((src_gray.shape[0]*src_gray.shape[1], 1))
src_new = np.float32(src_new)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
flags = cv2.KMEANS_PP_CENTERS
# 标签0为肌肉， 标签1为背景， 标签2为脂肪
compactness, labels, centers = cv2.kmeans(src_new, 3, None, criteria, 10, flags)
# 提取肌肉聚类数据
beefModel = np.where(labels == 0, 255, 0).astype("uint8")
# 提取脂肪聚类数据
fatModel = np.where(labels == 2, 255, 0).astype("uint8")
# 提取背景聚类数据
bgModel = np.where(labels == 1, 255, 0).astype("uint8")
# 肌肉数据成图
dst = beefModel.reshape((src.shape[0], src.shape[1]))
# 背景数据成图
background = bgModel.reshape((src.shape[0], src.shape[1]))
# 脂肪数据成图
fat = fatModel.reshape((src.shape[0], src.shape[1]))
# 分离大块的脂肪结蹄组织
# # 连通区域算法
image_label = label(fat, connectivity=2)
# 统计各脂肪块的大小
d = {}
for i in xrange(image_label.max() + 1):
    d[i] = np.sum(image_label == i)
# 进行排序
sort_label = sorted(d.iteritems(), key=lambda x: x[1], reverse=True)
# 设定肌间脂肪的极值
fat_excess = 3000
# print sort_label
# 创建一张图片，存放最大脂肪块
fat_max = np.zeros((src_gray.shape)).astype("uint8")
# 创建一张图片，存放非结蹄组织的脂肪块
img_fat = np.zeros((src_gray.shape[0], src_gray.shape[1]))
for row in xrange(fat.shape[0]):
    for col in xrange(fat.shape[1]):
        # 如果某点的像素与最大脂肪块的像素相同，将它提取到fat_max图像中
        if image_label[row, col] == sort_label[1][0]:
            fat_max[row, col] = 255
# 判断最大脂肪块是否是结蹄组织
if sort_label[1][1] >= fat_excess:
    for row in xrange(fat.shape[0]):
        for col in xrange(fat.shape[1]):
            if fat[row, col] < fat_max[row, col]:
                img_fat[row, col] = 0
            elif fat[row, col] >= fat_max[row, col]:
                img_fat[row, col] = fat[row, col] - fat_max[row, col]
else:
    for row in xrange(fat.shape[0]):
        for col in xrange(fat.shape[1]):
            img_fat[row, col] = fat[row, col]

cv2.imshow("dst", dst)
cv2.imshow("background", background)
cv2.imshow("fat", fat)
cv2.imshow("fat_max", fat_max)
cv2.imshow("fat_result", img_fat)
cv2.waitKey(0)




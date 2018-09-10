#! /usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

a = np.array([[1,0,1,1,0,1],
              [0,1,1,0,1,1],
              [0,0,1,1,0,1],
              [1,0,1,0,1,1]])
c = np.array([[1, 0, 1, 1],
              [0, 1, 1, 0],
              [0, 0, 1, 1],
              [1, 0, 1, 0]])
d = {"max_fat": 0, "min_fat": 0}
# print d["max_fat"]
rows = a.shape[0]
cols = a.shape[1]
px = 0
b = np.zeros((a.shape))


for row in xrange(a.shape[0]):
    for col in xrange(a.shape[1]):
        if a[row, col] == 1:
            if col+1 < cols and a[row, col] == a[row, (col+1)]:

                        b[row, col] = a[row, col]
                        b[row, (col+1)] = a[row, (col + 1)]

            if row + 1 < rows and a[row, col] == a[(row + 1), col]:

                        b[row, col] = a[row, col]
                        b[(row + 1), col] = a[(row + 1), col]


for row in xrange(a.shape[0]):
    for col in xrange(a.shape[1]):
        # print b[row, col]
        if row + 1 < rows and b[row, col] == 1:
            if b[row, col] != 1:
                print [row, col]

# print px


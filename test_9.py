#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
src = cv2.imread("head.jpg")
dst = src[36:232, 43:285]
cv2.namedWindow("output")
cv2.imshow("output", dst)
cv2.waitKey(0)

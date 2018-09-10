#! /usr/bin/env python
# -*- coding:utf-8 -*-
# 手工瞄点
import numpy as np
import cv2
Color = [0, 0, 255]
drawing = False
ix, iy = -1, -1

def onmouse(event,x,y,flags,param):
    global img, img2, ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 0, Color, 3)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = True
        ix, iy = x, y
        cv2.circle(img2, (ix, iy), 0, (0, 0, 0), 1)
        print ix, iy

if __name__ == "__main__":
    filename = "head.jpg"
    img = cv2.imread(filename)
    # img = cv2.resize(img, (img.shape[1]/4, img.shape[0]/4))
    img2 = np.full(img.shape, 255, np.uint8)
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)

    while (1):
        cv2.imshow('output', img2)
        cv2.imshow('input', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('result_output.png', img2)
            d = {}
            d["x"] = []
            d["y"] = []
            src = cv2.imread("result_output.png", 0)
            for row in xrange(src.shape[0]):
                for col in xrange(src.shape[1]):
                    if src[row, col] == 0:
                        d["x"].append(row)
                        d["y"].append(col)
            print d
    cv2.destroyAllWindows()


#! /usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt


'''
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]


# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
'''

'''
src = cv2.imread("C:/Users/12914/Pictures/demo.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
b = src[:,:,0]
g = src[:,:,1]
r = src[:,:,2]
plt.subplot(2,2,1)
plt.hist(src_gray.ravel(), 256), plt.title("original")
plt.subplot(2,2,2)
plt.hist(b.ravel(), 256), plt.title("blue")
plt.subplot(2,2,3)
plt.hist(g.ravel(), 256), plt.title("green")
plt.subplot(2,2,4)
plt.hist(r.ravel(), 256), plt.title("red")
plt.show()
'''






#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
src = cv2.imread("C:/Users/12914/Pictures/yanji.jpg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_new = src_gray.reshape((src_gray.shape[0]*src_gray.shape[1], 1))
z = np.float32(src_new)
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,3,None,criteria,10,flags)
A = z[labels==0]
B = z[labels==1]
C = z[labels==2]
# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
sns.set_style("darkgrid")
plt.hist(A,32,[0,256],color = 'r',label="Fat")
plt.hist(B,32,[0,256],color = 'b',label="muscle")
plt.hist(C,32,[0,256],color = 'g',label="background")

#plt.legend(bbox_to_anchor=(0,1.02,1,0.102), loc=3, ncol=3, mode='expand',borderaxespad=0)
plt.legend(loc=1, ncol=1)
plt.xlabel("Pixel")
plt.ylabel("Count")
plt.title("K-Means Clustering")
plt.savefig("K-Means Clustering.png")


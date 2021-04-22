# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:30:41 2017

@author: kel, gunjan
"""
import pandas as pa
import scipy.spatial.distance as spa
from scipy.sparse.csgraph import dijkstra
import scipy as sp
import random
import math
from repo.dimredu import MCviaIALMFast

import numpy as np
from sklearn.neighbors import BallTree
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt
import networkx as nx
plt.close('all')

print('loading --------------------')

import os
path = ("C:/Users/Puzzle Assembly/Desktop/other files/precipitate")
print (path)


# exit()


# path = 'files'

#  load original TPM for reference

original = np.loadtxt('nw_0.txt')
# odd = np.loadtxt('odd_0.txt')

nw_5 = np.loadtxt('nw_5.txt')
nw_10 = np.loadtxt('nw_10.txt')
nw_20 = np.loadtxt('nw_20.txt')
nw_40 = np.loadtxt('nw_40.txt')
nw_60 = np.loadtxt('nw_60.txt')
nw_80 = np.loadtxt('nw_80.txt')


r,c = original.shape


print('loading done--------------------')

# exit()
# print(len(s1))
# print(len(x))

# print(s1)
# print(np.log(s1))

plt.figure(1)

# linearize data


hop1 = np.zeros(((r*c)))
hop2 = np.zeros(((r*c)))
hop3 = np.zeros(((r*c)))
hop4 = np.zeros(((r*c)))
hop5 = np.zeros(((r*c)))
hop6 = np.zeros(((r*c)))
ori = np.zeros(((r*c)))

p = 0
for i in range(r):
  for j in range(c):
    ori[p] = (original[i,j])
    hop1[p] = (nw_5[i,j]) 
    hop2[p] = (nw_10[i,j])
    hop3[p] = (nw_20[i,j])
    hop4[p] = (nw_40[i,j])
    hop5[p] = (nw_60[i,j])
    hop6[p] = (nw_80[i,j])
    p = p+1

# -------------------------- nw1 -----------------------------------------

print ('--nw1--')
print(r,c)


# x = mean error = percentage error 

m5 = []
m10 = []
m20 = []
m40 = []
m60 = []
m80 = []

abs_ori = np.copy(ori)
abs_ori = np.abs(abs_ori)

# for 5% deletion
x = np.round(hop1-ori)
x = np.abs(x)
x20 = (np.sum(abs(x)))/(np.sum(abs_ori))
x20 = x20*100
y20 = (np.sum(abs(x)))/(r*c)
e20 = np.std(abs(x))
hop_1_20 = abs(x)

# for 10% deletion
x = np.round(hop2-ori)
x40 = (np.sum(abs(x)))/(np.sum(abs_ori))
x40 = x40*100
y40 = (np.sum(abs(x)))/(r*c)
e40 = np.std(abs(x))
hop_1_40 = abs(x)

# for 20% deletion
x = np.round(hop3-ori)
x60 = (np.sum(abs(x)))/(np.sum(abs_ori))
x60 = x60*100
y60 = (np.sum(abs(x)))/(r*c)
e60 = np.std(abs(x))
hop_1_60 = abs(x)

# for 40% deletion
x = np.round(hop4-ori)
x80 = (np.sum(abs(x)))/(np.sum(abs_ori))
x80 = x80*100
y80 = (np.sum(abs(x)))/(r*c)
e80 = np.std(abs(x))
hop_1_80 = abs(x)

# for 60% deletion
x = np.round(hop5-ori)
x90 = (np.sum(abs(x)))/(np.sum(abs_ori))
x90 = x90*100
y90 = (np.sum(abs(x)))/(r*c)
e90 = np.std(abs(x))

# for 80% deletion
x = np.round(hop6-ori)
x0 = (np.sum(abs(x)))/(np.sum(abs_ori))
x0 = x0*100
y0 = (np.sum(abs(x)))/(r*c)
e0 = np.std(abs(x))



print('---x20---',x20)


# exit()

# make an array of mean errors
line100 = []
line100 = np.insert(line100, 0, x90)
line100 = np.insert(line100, 0, x80)
line100 = np.insert(line100, 0, x60)
line100 = np.insert(line100, 0, x40)
line100 = np.insert(line100, 0, x20)
line100 = np.insert(line100, 0, 0)

# make an array of deviation in errors
dev1 = []
dev1 = np.insert(dev1, 0, e90)
dev1 = np.insert(dev1, 0, e80)
dev1 = np.insert(dev1, 0, e60)
dev1 = np.insert(dev1, 0, e40)
dev1 = np.insert(dev1, 0, e20)
dev1 = np.insert(dev1, 0, 0)

print('mean:',line100)
print('dev:',dev1)

np.savetxt('mean_err.txt',line100)
np.savetxt('dev_err.txt',dev1)

x = [0, 20, 40, 60, 80, 90]
plt.figure(1)

# ------------to plot without error bars---------------------------------------------------------------------------------------

# plt.plot( x,line100, 'bo-', markeredgecolor='black', label='Facebook Network - 50 anchors')
# plt.plot( x,line150, 'y^-', markeredgecolor='black', label='Facebook Network - 100 anchors')
# plt.plot( x,line200, 'rs-', markeredgecolor='black', label='Facebook Network - 150 anchors')

# ------------to plot error bars---------------------------------------------------------------------------------------

plt.errorbar(x, line100, dev1, linestyle='-', marker='o',markerfacecolor ='blue', label='Circular Network with three voids')
# plt.errorbar(x, line150, dev2, linestyle='-', marker='^',markerfacecolor ='yellow', label='Facebook Network - 100 anchors ')
# plt.errorbar(x, line200, dev3, linestyle='-', marker='s',markerfacecolor ='red', label='Facebook Network - 150 anchors')


# plt.title('Mean error - Facebook network - different anchors')
plt.xlabel('(%) Missing Coordinates in VC Matrix')
plt.ylabel('Mean Error (%)')
plt.legend(loc='upper left') 



plt.margins(x=0)

plt.show()


# plt.title('hop histogram')

# # ------------------------------------------------------
# # Histogram of hop distances nw1
# # plt.figure(1)
# plt.hist(hop1,20,alpha=0.5,label='20% distance matrix entries removed')
# # plt.xlabel('Hop Distance')
# # plt.ylabel('Number of Distance Matrix Entries')

# # plt.figure(2)
# plt.hist(hop2,40,alpha=0.5,label='40% distance matrix entries removed')
# # plt.xlabel('Hop Distance')
# # plt.ylabel('Number of Distance Matrix Entries')

# # plt.figure(3)
# plt.hist(hop3,20,alpha=0.5,label='60% distance matrix entries removed')
# plt.xlabel('Hop Distance')
# plt.ylabel('Number of Distance Matrix Entries')

# plt.hist(hop4,20,alpha=0.5,label='80% distance matrix entries removed')
# #--------------------------------------------------------------------------

# plt.legend(loc='upper right')


# exit()

# absolute hop distance error with std. deviation: (MEAN ERROR) for nw3 with anchors (MC code)
# x20 = (np.sum(hop1-ori))/(r1*c1)
# x40 = (np.sum(hop2-ori))/(r2*c2)
# x60 = (np.sum(hop3-ori))/(r3*c3)
# x80 = (np.sum(hop4-ori))/(r4*c4)
# x90 = (np.sum(hop5-ori))/(r5*c5)

# df = pa.DataFrame({"nw1_mean" : line100, "nw2_mean" : line150, "nw3_mean" : line200,})
# df.to_csv("nw1_nw2_nw3_with_bounds_mean_error.csv", index=False, header=False)


# df = pa.DataFrame({"nw1_hop" : line1, "nw1_stddev" : dev1, "nw2_hop" : line2, "nw2_stddev" : dev2, "nw3_hop" : line3, "nw3_stddev" : dev3})
# df.to_csv("with_bounds_nw1_hop_error_stddev_nw2_hop_std_nw3_hop_std.csv", index=False, header=False)




plt.show()

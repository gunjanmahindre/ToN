# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:30:41 2017

@author: gunjan,kel
"""

import timeit
start_time = timeit.default_timer()


import numpy as np
#from sklearn.neighbors import BallTree
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt

import pandas as pa
import scipy.spatial.distance as spa
from scipy.sparse.csgraph import dijkstra
import scipy as sp
import random
import math
from dimredu import MCviaIALMFast
from dimredu import sRPCAviaADMMFast

import networkx as nx

plt.close('all')


n = 0
col_size = 0
row_size = 0



def rand_removal(VC,C,fraction):

  [Rn, Cn] = VC.shape
  rem_num = (Rn*Cn*fraction/100) # total number of entries to be removed
  rem_num = int(rem_num) 

  VCcopy = np.copy(VC)
  print (rem_num)

  RR = np.random.choice(Rn, rem_num)
  CC = np.random.choice(Cn, rem_num)

  VCr = np.copy(VC)
  for xx in range(rem_num):
    i=RR[xx]
    j=CC[xx]
    VCr[i][j] = 0
    VCcopy[i][j] = -1
  return(VCr,rem_num,VCcopy)



## start of code:------------------------

flag = 0 #0 for random or 1 for ens
fraction = 0 #percentage of elements to be removed

#the below is always 0
# removal = 0    #0 for random removal of elements 1 for removing last set of elements
# lets just do random removal


# var = np.loadtxt('spiral.txt',delimiter='\t')
# var = np.loadtxt('odd_shaped.txt',delimiter='\t')
var = np.loadtxt('three-void.txt',delimiter='\t')
# 




D = spa.pdist(var,'euclidean')

#gives Distance matrix in square form
sqr = spa.squareform(D)
#defining radius of connection of a node: r
r = 1
#extracting connectivity matrix = adj as per the connetivity
adj =+ spa.squareform(D<=r)
#stores number of nodes in 'n'
n = len(var)
row_size = n
#convert the adj matrix into sparse form
sps = sp.sparse.coo_matrix(adj)
#create a complete distance matrix
A = np.array(dijkstra(sps))

##select random anchors from VC matrix = A
# rand_ens(A)

col_size = 20
if fraction == 0:
  C = np.array(random.sample(range(0,row_size),col_size))
  np.save('random_anchors.npy',C)
else:
  C = np.load('random_anchors.npy')


# ----
# C = np.load('odd_random_anchors.npy')
# C = np.load('three_random_anchors.npy')
col_size = (len(C))
VC =  A[:,C]
# print (VC[0,:])
# exit()


[VC_removed, removed, VCc] = rand_removal(VC,C,fraction)
# print (VC_removed)
# print ("total removed entries: ", removed)
# print ("fraction: ", fraction)




# ----------------------------------calling Matrix Completion----------
 
m,n = VC_removed.shape
# print ('m,n=',m,n)
# create u,v and vecM vectors
# u = np.zeros(1)
# v = np.zeros(1)
# vecM = np.zeros(1)
u = []
v = []
vecM = []
for i in range(m):
  for j in range(n):
    if VCc[i,j]!=(-1):
      u.append(i)
      v.append(j)
      vecM.append(VCc[i,j])
# print (VCc)
maxRank = col_size
u = np.asarray(u)

# print ('*******************')

v = np.asarray(v)
vecM = np.asarray(vecM)
# print (len(u))
# print (len(v))
# print (len(vecM))
# print (np.where(u>495))
# print (u)
# print (v)
# print (vecM)

# print (type(u))
# print (type(v))
# print (type(vecM))

# print (VC_removed[0,0])

# exit()

U, S, VT = MCviaIALMFast.MC(m,n,u,v,vecM,maxRank)
# RCP: I added some tests to make sure that S is not used
# assert np.all(S.todense()==0),'****DANGER****'
VT = VT.transpose()
Psvd = np.dot(U,np.diag(S))
# print (Psvd.shape)
Psvd = np.array(Psvd)
# plt.figure(1)
# plt.scatter(Psvd[:,1],Psvd[:,2])

# plt.show()

dc = np.zeros( (len(Psvd), 2) )
dc[:,0] = Psvd[:,1]
dc[:,1] = Psvd[:,2]

np.savetxt('nw_0.txt', dc, delimiter = '\t')

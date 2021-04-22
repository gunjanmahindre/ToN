

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
from itertools import permutations

from mpl_toolkits.mplot3d import axes3d, Axes3D 

import os
plt.close('all')


# XXX = np.load('T_coor.npy')
# np.savetxt('t1.csv',XXX,delimiter=',')
# # np.savetxt('h1.txt',XXX[:,0])
# # np.savetxt('h2.txt',XXX[:,1])
# # np.savetxt('h3.txt',XXX[:,2])

# exit()

# XXX = np.loadtxt('T_shaped_3D.npy',delimiter='\t')
XY = np.load('hourglass_coor.npy')
YX = np.loadtxt('YYY_hourglass.txt',delimiter='\t')
ZZ = np.loadtxt('ZZZ_hourglass.txt',delimiter='\t')

# XY = np.loadtxt('three-void.txt',delimiter='\t')
# XY = np.loadtxt('odd_shaped.txt',delimiter='\t')
# YX = np.loadtxt('YX_three.txt',delimiter='\t')
# YX = np.loadtxt('YX_odd.txt',delimiter='\t')

#  load double centered xy coordinates:
# Psvd = np.load(path+'/coors_dc.npy')
# Psvd = np.loadtxt('odd_coors_svd_dc_80.txt') 
Psvd = np.loadtxt('h_10.txt') 

# print (Psvd.shape)

# exit()



# -----------orient SVD in YYY domain-----------------
[nodes, c] = XY.shape
order2 = []
for i in range (nodes):
	# print (XY[i,:])
	loc1 = np.where(XY[:,0] == YX[i,0])
	# print (loc1)
	loc2 = np.where(XY[:,1] == YX[i,1])
	# print (loc2)
	loc = np.intersect1d(loc1,loc2)
	# print (loc)
	order2.insert(len(order2), loc)

order2 = np.asarray(order2)
order = []
for i in range (nodes):
	order.insert(len(order) , order2[i][0])
	# print (i, order[i])
order = np.asarray(order)

# ----------------------------------------------------

# np.savetxt('XYsort.csv', XY, delimiter='\t')
# np.savetxt('Psvd_sort.csv', Psvd, delimiter='\t')

# print (Psvd.shape)


txt = np.arange(nodes)

print (txt.shape)

# exit ()
# --------------------plot original -----------------
fig = plt.figure()
ax = Axes3D(fig)

# ax.tick_params(axis='both', which='major', labelsize=12)
ax.scatter(Psvd[:,0],Psvd[:,1],Psvd[:,2], c=Psvd[:,2])
# ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plt.show()

# exit()
# -----------------------------------------here


# plt.figure(1)
# plt.plot(Psvd[:,0], Psvd[:,1])
# for i in txt:
# 	s = str(txt[i])
# 	plt.text(Psvd[i,0], Psvd[i,1], s) 


# plt.figure(2)
# plt.plot(XY[:,0], XY[:,1])
# for i in txt:
# 	s = str(txt[i])
# 	plt.text(XY[i,0], XY[i,1], s) 

# plt.show()

# exit()

planes = []
all_planes = len(XY[:,2])

for pp in range (10):
	# TPM errro in Y direction:
	I = 0
	nn = 1  #nodes in the current line
	Y_err = []
	Y_perm = []

	for i in range(nodes-1):
		j=i+1
		if XY[i,0]==XY[j,0]: #check if we are on the same line
			# increase the node number
			nn = nn+1
			if Psvd[i,0]>Psvd[j,0]:  # check if there is an error
				# increase error count
				I = I+1
			else:
				# calculate error for this line		
				upper = math.factorial(nn)
				lower = math.factorial(nn-2)
				perm = upper/lower
				# print ('I',I)
				# print ('nn',nn)
				Y_err.insert(len(Y_err), I)
				Y_perm.insert(len(Y_perm), perm)
				# print (Y_tpm)
				# reset nn and I error value, start new line
				I = 0
				nn = 1
	print ('total error in Y direction (%) :')
	up = sum(Y_err)
	print (up)
	down = sum(Y_perm)
	print (down)
	# total Y error for this plane
	Yerr = (up/down)*100
	print (Yerr)
	planes.insert(len(planes), Yerr)

plt.show()

exit()

# TPM errro in x direction:---------------------------------------------

[nodes, c] = YX.shape
txt = np.arange(nodes)


plt.figure(3)
plt.plot(Psvd[:,0], Psvd[:,1])
for i in order:
	s = str(order[i])
	plt.text(Psvd[i,0], Psvd[i,1], s) 


plt.figure(4)
plt.plot(YX[:,0], YX[:,1])
for i in txt:
	s = str(txt[i])
	plt.text(YX[i,0], YX[i,1], s) 

# plt.show()



I = 0
nn = 1  #nodes in the current line
X_err = []
X_perm = []
# print (nodes)

for i in range(nodes-1):
	# print ('checking if they are on the same line')
	j=i+1
	if YX[i,1]==YX[j,1]: #check if we are on the same line
		# increase the node number
		# print ('they are on the same Y level')
		nn = nn+1
		# print (i,j)
		
		p1 = order[i]
		p2 = order[j]

		if Psvd[p1 ,1]>Psvd[p2 ,1]:  # check if there is an error
			# increase error count
			I = I+1
			print (i,j)
			print (Psvd[p1,1], Psvd[p2,1])
		else:
			
			# calculate error for this line		
			upper = math.factorial(nn)
			lower = math.factorial(nn-2)
			perm = upper/lower
			# print ('I',I)
			# print ('nn',nn)
			X_err.insert(len(X_err), I)
			X_perm.insert(len(X_perm), perm)
			# print (Y_tpm)
			# reset nn and I error value, start new line
			I = 0
			nn = 1


print ('total error in X direction (%) :')

up = sum(X_err)
down = sum(X_perm)

Xerr = (up/down)*100
print (Xerr)


print ('total TPM error:')
up = sum(X_err) + sum(Y_err)
down = sum(X_perm) + sum(Y_perm)

total_tpm = (up/down)*100
print (total_tpm)


print ('------------------check------------------')
print (sum(X_err) , sum(X_perm))
print (sum(Y_err) , sum(Y_perm))
print (up, down)


print ('-------------final X Y and total')
print (Xerr)
print (Yerr)
print (total_tpm)

# ALLerr = []
# ALLerr.insert(len(ALLerr), Xerr)
# ALLerr.insert(len(ALLerr), Yerr)
# ALLerr.insert(len(ALLerr), total_tpm)


# np.savetxt('10_1.txt', ALLerr)

plt.show()


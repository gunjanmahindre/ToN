# calls dc_tpm_error

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

import os



# print('loading --------------------')

import os
path = ("C:/Users/Puzzle Assembly/Desktop/other files/precipitate")
# print (path)


# exit()


# path = 'files'

#  load original TPM for reference
# original = np.loadtxt('nw2_0.txt')







def dc_tpm_error(fraction,nw):
	XY = np.loadtxt('three-void.txt',delimiter='\t')
	# XY = np.loadtxt('odd_shaped.txt',delimiter='\t')
	YX = np.loadtxt('YX_three.txt',delimiter='\t')
	# YX = np.loadtxt('YX_odd.txt',delimiter='\t')

	#  load double centered SVD xy coordinates:

	# Psvd = np.loadtxt('nw2_80.txt') 
	Psvd = nw

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


	# order2 = np.transpose(order2)

	order = []
	for i in range (nodes):
		order.insert(len(order) , order2[i][0])
		# print (i, order[i])

	order = np.asarray(order)

	# print (order[0:10])

	# exit()


	# np.savetxt('XYsort.csv', XY, delimiter='\t')
	# np.savetxt('Psvd_sort.csv', Psvd, delimiter='\t')

	# print (Psvd.shape)


	txt = np.arange(nodes)

	# print (txt.shape)

	# exit ()

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


	# TPM errror in Y direction:
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

	# print ('total error in Y direction (%) :')

	up = sum(Y_err)
	down = sum(Y_perm)
	Yerr = (up/down)*100
	# print (Yerr)


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
				# print (i,j)
				# print (Psvd[p1,1], Psvd[p2,1])
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


	# print ('total error in X direction (%) :')

	up = sum(X_err)
	down = sum(X_perm)

	Xerr = (up/down)*100
	# print (Xerr)


	# print ('total TPM error:')
	up = sum(X_err) + sum(Y_err)
	down = sum(X_perm) + sum(Y_perm)

	total_tpm = (up/down)*100
	# print (total_tpm)


	# print ('------------------check------------------')
	# print (sum(X_err) , sum(X_perm))
	# print (sum(Y_err) , sum(Y_perm))
	# print (up, down)


	print ('-------------final X Y and total')
	print (Xerr)
	print (Yerr)
	print (total_tpm)
	return(Xerr, Yerr, total_tpm)

	# ALLerr = []
	# ALLerr.insert(len(ALLerr), Xerr)
	# ALLerr.insert(len(ALLerr), Yerr)
	# ALLerr.insert(len(ALLerr), total_tpm)


	# np.savetxt('10_1.txt', ALLerr)



	# plt.show()

Xerr = []
Yerr = []
total_tpm = []
	
# nw = np.loadtxt('nw_10.txt')
# fraction = 10
# print (fraction)
# x, y, t = dc_tpm_error(fraction,nw)
# Xerr.insert(len(Xerr), x)
# Yerr.insert(len(Yerr), y)
# total_tpm.insert(len(total_tpm), t)

# # exit()

nw = np.loadtxt('nw_0.txt')
fraction = 0
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

nw = np.loadtxt('nw_10.txt')
fraction = 10
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

nw = np.loadtxt('nw_20.txt')
fraction = 20
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

nw = np.loadtxt('nw_40.txt')
fraction = 40
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

nw = np.loadtxt('nw_60.txt')
fraction = 60
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

nw = np.loadtxt('nw_80.txt')
fraction = 80
print (fraction)
x, y, t = dc_tpm_error(fraction,nw)
Xerr.insert(len(Xerr), x)
Yerr.insert(len(Yerr), y)
total_tpm.insert(len(total_tpm), t)

# ----save x error---------------
# hs = open("tpm_x.txt","a")
# hs.write(str('Iteration_12'))

# for i in range(7):
# 	# hs = open("tpm_x.txt","a")
# 	hs.write('\n')
# 	hs.write(format(fraction))
# 	hs.write('%')
# 	hs.write(' ')
# 	hs.write(format(Xerr[i]))
# hs.write('\n')
# hs.write('\n')
# hs.close() 

# # ----save y error---------------
# hs = open("tpm_y.txt","a")
# hs.write(str('Iteration_12'))

# for i in range(7):
# 	# hs = open("tpm_y.txt","a")
# 	hs.write('\n')
# 	hs.write(format(fraction))
# 	hs.write('%')
# 	hs.write(' ')
# 	hs.write(format(Yerr[i]))
# hs.write('\n')
# hs.write('\n')
# hs.close() 

# ----save total error---------------

x = [0, 10, 20, 40, 60, 80]

hs = open("tpm_total.txt","a")
hs.write(str('Iteration_12'))

for i in range(6):
	# hs = open("tpm_total.txt","a")
	hs.write('\n')
	hs.write(format(x[i]))
	hs.write('%')
	hs.write(' ')
	hs.write(format(total_tpm[i]))
hs.write('\n')
hs.write('\n')
hs.close() 
	

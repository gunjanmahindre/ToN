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



## start of code:------------------------


#the below is always 0
# removal = 0    #0 for random removal of elements 1 for removing last set of elements
# lets just do random removal


# var = np.loadtxt('spiral.txt',delimiter='\t')
# var = np.loadtxt('odd_shaped.txt',delimiter='\t')
var = np.loadtxt('t_pipe.txt',delimiter='\t')
D = spa.pdist(var,'euclidean')

#gives Distance matrix i`n square form
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
dHp = np.array(dijkstra(sps))




np.savetxt('t_pipe_Dist.txt',dHp)
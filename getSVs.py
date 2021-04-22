import pandas as pa
import scipy.spatial.distance as spa
from scipy.sparse.csgraph import dijkstra
import scipy as sp
import random
import math

import numpy as np
from sklearn.neighbors import BallTree
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt
import networkx as nx

path = "C:\\SkyDrive\\PhD new\\Projects\\ToN\\codes\\"

dHp = np.loadtxt(path+'three_void_Dist.txt')
row_size = len(dHp)
col_size = 20
C = np.array(random.sample(range(0,row_size),col_size))
nw1 =  dHp[:,C]

dHp = np.loadtxt(path+'odd_shaped_Dist.txt')
row_size = len(dHp)
col_size = 20
C = np.array(random.sample(range(0,row_size),col_size))
nw1 =  dHp[:,C]

dHp = np.loadtxt(path+'hourglass3D_Dist.txt')
row_size = len(dHp)
col_size = 20
C = np.array(random.sample(range(0,row_size),col_size))
nw1 =  dHp[:,C]

dHp = np.loadtxt(path+'t_pipe_Dist.txt')
row_size = len(dHp)
col_size = 20
C = np.array(random.sample(range(0,row_size),col_size))
nw1 =  dHp[:,C]


# plot singular values:
u1,s1,v1 = np.linalg.svd(nw1)
u2,s2,v2 = np.linalg.svd(nw2)
u3,s3,v3 = np.linalg.svd(nw3)
u4,s4,v4 = np.linalg.svd(nw4)

x1 = np.arange(0,len(s1),1)
x2 = np.arange(0,len(s2),1)
x3 = np.arange(0,len(s3),1)
x4 = np.arange(0,len(s4),1)



plt.figure(1)
# 'Collaboration Network - 100 anchors'
plt.plot( np.log(s1), 'bo-', markeredgecolor='black', label='Circular network with three voids')
plt.plot( np.log(s2), 'y^-', markeredgecolor='black', label='Odd shaped network')
plt.plot( np.log(s3), 'r.-', markeredgecolor='black', label='Cube with hourglass shaped void network')
plt.plot( np.log(s4), 'g--', markeredgecolor='black', label='Hollow T shaped cylinder network')

plt.ylabel('Singular Values(natural log scale)')
plt.xlabel('Component Number')
plt.legend(loc='upper right') 

plt.show()
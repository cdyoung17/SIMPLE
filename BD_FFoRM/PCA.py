#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import numpy as np
import pickle

import scipy.io
from scipy import signal
from sklearn.utils.extmath import randomized_svd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as mtri
from matplotlib import cm

mymap=matplotlib.cm.get_cmap('jet')

# trunc = int(sys.argv[2])
# win = int(sys.argv[1])
load = int(sys.argv[1])

step_size = 0.05
# Sweep parameter - flow rate ratio controls flow type, ext vs shear
FRR = np.arange(-0.7,1.00001,step_size)
# FRR = [0.05]
FRR = FRR[::2]
nF = len(FRR)
nQ = 1
# Boundary conditions - inlet flow rates
# Q2 = np.zeros((nF,nQ))
Q2max = 1.0e-6
Q2min = 1.0e-8
# Q2[:] = np.linspace(Q2min,Q2max,nQ)
Q2arr = np.linspace(Q2min,Q2max,10)
Q2 = Q2arr[1::2]
# Q2 = np.array([3.4e-7])
print(Q2)
# Q2[:] = [1.2e-7]
Q2 = Q2/1.0e-7
strm = np.arange(0,10)
# strm = np.array([4])
avg_win = 0.01
nqi = 100
qlim = 0.1
# nthph = 300
nthph = 200
npsi = 20
ntraj = 100000

L = 1680.0
R = 34.0
volfrac=0.1 # volume fraction of cylinders
dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
bkgd = 0.0001 # background scattering in cm^-1
data_path = 'training_output/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nthph,nqi,qlim,L,R)

n_load = 106663
Iq = pickle.load(open(data_path+'IQphase_snapshots_clearnan_%d.p'%n_load,'rb'))

# Test-train split
M = len(Iq)
# frac=.8
frac = 1
Iq_train=Iq[:round(M*frac),:]
Iq_test=Iq[round(M*frac):M,:]

# Transpose for SVD
Iq_train = np.transpose(Iq)
Iq_test = np.transpose(Iq)

# max_trunc = Iq.shape[1]
max_trunc = 100
print(max_trunc)

if load==0:
	U,S,VT=randomized_svd(Iq_train, n_components=max_trunc)
	pickle.dump([U,S],open(data_path+'Qphase_pca_%d.p'%(len(Iq)),'wb'))

elif load==1:
	U,S = pickle.load(open(data_path+'Qphase_pca_%d.p'%n_load,'rb'))

trunc = 50
alpha = Iq @ U[:,:trunc]
print('reduced order model shape',alpha.shape)
pickle.dump(alpha,open(data_path+'red_aligned_snaps_clearnan_%d_%d.p'%(trunc,n_load),'wb'))

print('U shape',U.shape)
print('S shape',S.shape)
Iq_pred = np.transpose(alpha @ np.transpose(U[:,:trunc]))
print('Iq_pred shape',Iq_pred.shape)
mean = np.mean(Iq_train)
err = np.mean(np.sqrt((Iq_train - Iq_pred)**2))/mean
print('Normalized reconstruction error %.3e'%err)
exit()

# Code for calculating MSE, SVs vs mode number below

#Calculate coefficients for the PCA modes
# alpha=np.transpose(ulinear_test) @ U[:,:trunc]
# alpha = np.transpose(Iq_test) @ U
# # Recreate test data based on trunc modes
# Iq_pred = np.transpose(alpha @ np.transpose(U))

# mse = np.mean((Iq_test - Iq_pred)**2)
# print('%d PCA modes mse %.3e'%(trunc,mse))
# # print(mse)

# max_trunc = 50

# for tru in range(1,max_trunc+1):

# 	alpha = Iq @ U[:,:tru]
# 	print('reduced order model shape',alpha.shape)

# 	pickle.dump(alpha,open(data_path+'r_red_%d_PCA_modes_%d.p'%(tru,n_load),'wb'))

# exit()

SV = np.sqrt(S)
# print(S)
# print(len(SV))

fig = plt.figure(figsize=(4,3),dpi=200)
ax = fig.add_subplot(111)

plt.loglog(np.arange(1,len(SV)+1),SV)
plt.xlabel('Singular value')
plt.tight_layout()
# plt.show()
plt.savefig(open('Qphase_singular_values_%d.png'%len(Iq),'wb'))

# max_trunc = 500
mse = np.zeros(max_trunc)

for tru in range(1,max_trunc+1):
	alpha = np.transpose(Iq_test) @ U[:,:tru]
	# print(alpha.shape)
	# Recreate test data based on trunc modes
	Iq_pred = np.transpose(alpha @ np.transpose(U[:,:tru]))

	mse[tru-1] = np.mean((Iq_test - Iq_pred)**2)
	print('%d PCA modes mse %.3e'%(tru,mse[tru-1]))
	# print(mse)

pickle.dump(mse,open('Qphase_mse_%d.p'%len(Iq),'wb'))

fig = plt.figure(figsize=(4,3),dpi=200)
ax = fig.add_subplot(111)

plt.semilogy(np.arange(1,max_trunc+1),mse)
plt.xlabel('# of PCA modes')
plt.ylabel('MSE')
plt.tight_layout()
# plt.show()
plt.savefig(open('Qphase_mse_%d.png'%len(Iq),'wb'))

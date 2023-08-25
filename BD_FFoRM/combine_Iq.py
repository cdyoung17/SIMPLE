#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import numpy as np
import pickle
import pandas as pd

# This code combines data from the individual trajectories into a single file for loading into ML training codes

step_size = 0.05
# Sweep parameter - flow rate ratio controls flow type, ext vs shear
FRR = np.arange(-0.7,1.00001,step_size)
FRR = FRR[::2]
# FRR = [-0.7]
# nF = len(FRR)
nQ = 10
# Boundary conditions - inlet flow rates
#Q2 = np.zeros((nF,nQ))
Q2max = 1.0e-6
Q2min = 1.0e-8
# Q2 = np.linspace(Q2min,Q2max,nQ)
Q2arr = np.linspace(Q2min,Q2max,10)
Q2 = Q2arr[1::2]
# Q2 = np.array([3.4e-7])
# print(Q2)
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
data_path = 'preprocessed/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/training_data/'%(nthph,nqi,qlim,L,R)

count = 0
len_I = 0
I_app = []
Q_app = []
phase_t_app = []
Gdot_app = []
E_app = []
ali_stress_app = []
n_strm = len(FRR)*len(Q2)*len(strm)
print('No. streamlines',n_strm)
crossover_indices = np.zeros(n_strm,dtype=int)
for f in FRR:
	for q in Q2:
		for st in strm:

			I,Q,phase_t,Gdot,E,aligned_stress = pickle.load(open(data_path+'IQphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(f,q,st),'rb'))
			Q = np.reshape(Q,(Q.shape[0],Q.shape[1]*Q.shape[2]))

			len_I += len(I)
			print(count,len_I,len(I))
			# print(I_app.shape)
			crossover_indices[count] = len_I - 1

			If = pd.DataFrame(I)
			Qf = pd.DataFrame(Q)
			phase_tf = pd.DataFrame(phase_t)
			Gdotf = pd.DataFrame(Gdot)
			Ef = pd.DataFrame(E)
			sf = pd.DataFrame(aligned_stress)

			I_app.append(If)
			Q_app.append(Qf)
			phase_t_app.append(phase_tf)
			Gdot_app.append(Gdotf)
			E_app.append(Ef)
			ali_stress_app.append(sf)

			# print(I_app.shape)
			# print(Q_app.shape)
			# print(phase_t_app.shape)

			count = count + 1
			del I,If,Q,Qf,phase_t,phase_tf,Gdot,Gdotf,E,Ef,aligned_stress,sf

print('Crossover indices')
for i in range(n_strm):
	print(i,crossover_indices[i])

pickle.dump(crossover_indices,open(data_path+'../crossover_indices_%d.p'%len_I,'wb'))

I_app = pd.concat(I_app)
Q_app = pd.concat(Q_app)
phase_t_app = pd.concat(phase_t_app)
Gdot_app = pd.concat(Gdot_app)
E_app = pd.concat(E_app)
ali_stress_app = pd.concat(ali_stress_app)

I_app = np.asarray(I_app)
Q_app = np.asarray(Q_app)
phase_t_app = np.asarray(phase_t_app)
Gdot_app = np.asarray(Gdot_app)
E_app = np.asarray(E_app)
ali_stress_app = np.asarray(ali_stress_app)

print(I_app.shape)
# print(Q_app.shape)
# origin = int(nqi**2/2)
# print('origin index %d'%origin)
ind = np.argwhere(np.isnan(I_app[:,0]))
print(ind.shape)
ind = ind.flatten()
print(ind.shape)
print(ind)
pickle.dump(ind,open(data_path+'../snapshots_skip_nan_%d.p'%len(I_app),'wb'))
#for i in range(len(ind)):
#	print(i,ind[i])
I_app = np.delete(I_app,ind,axis=0)
# ornt_app = np.delete(ornt_app,ind)
ind = np.argwhere(np.isnan(I_app))
print('nan indices',ind)
print('phase shape, Q shape, Gdshape, Eshape',phase_t_app.shape,Q_app.shape,Gdot_app.shape,E_app.shape)

pickle.dump(I_app,open(data_path+'IQphase_snapshots_clearnan_%d.p'%len(I_app),'wb'))
pickle.dump([Q_app,phase_t_app],open(data_path+'Qphase_snaps_%d.p'%len(Q_app),'wb'))
pickle.dump([Gdot_app,E_app],open(data_path+'GdotE_snaps_%d.p'%len(Gdot_app),'wb'))
pickle.dump(ali_stress_app,open(data_path+'aligned_stress_snaps_%d.p'%len(ali_stress_app),'wb'))

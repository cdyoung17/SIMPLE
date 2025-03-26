#! /home/cyoung37/anaconda3/bin/python3
import os
import sys
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

dPCA = 50
# batch_time = np.array([1,2,3,5,10])
# batch_time = np.array([10])
t = 11
# niters = 4000
# niters = 40000/batch_time
alpha = np.array([0.0,1.0e-4,1.0e-3,1.0e-2])
arch = 0
traj = np.arange(10,13)
load = 0
# nodes = [43,45,46]
# ppn = [24,24,24]
# nnodes = len(nodes)

data_size = 95119
# wdir = 'l2c_smooth_sin'
wdir = 'l2c_ups_smooth'
dh = 5
ae_model = 0

# data_size = 95181
# wdir = 'l2c'
# dh = 8
# ae_model = 2

step_size = 0.05
FRR = np.arange(-0.7,1.00001,step_size)
FRR = FRR[::2]
print(FRR)
nF = len(FRR)
nQ = 1
Q2max = 1.0e-6
Q2min = 1.0e-8
Q2arr = np.linspace(Q2min,Q2max,10)
# Q2 = np.array([7.8e-7])
Q2 = Q2arr[2::2]
Q2 = Q2/1.0e-7
print(Q2)
strm = np.arange(0,20)
movie = 0

# movie = 1
# FRR = np.array([-0.7,-0.5,0.0,0.3,0.6,1.0])
# # Q2 = np.array([3.4,3.4,3.4,3.4,3.4])
# # strm = np.array([4,4,4,4,4])
# # Q2 = np.zeros(len(FRR)) + 10.0
# Q2 = np.array([1.2,3.4,10.0])
# # strm = np.zeros(len(FRR),dtype=int) + 4
# st = 4

count_threads = 0
node = 0
count = 0
inds = 0

al = 0.001
tr = 9

npts = 50
teval = np.arange(0,npts)/npts

# for al in alpha:
# 	for tr in traj:
count = 0
ddir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d/interp/'%(dh,dPCA,wdir,arch,t,al,tr)
ddir2 = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d/'%(dh,dPCA,wdir,arch,t,al,tr)
# os.chdir(ddir)
labt_avg = np.zeros(teval.shape)
alit_avg = np.zeros(teval.shape)
latt_avg = np.zeros(teval.shape)
pht_avg = np.zeros(teval.shape)
for f in FRR:
	for q in Q2:
		for st in strm:

			tspan,err_lab,err_ali,err_lat,err_ph = pickle.load(open(ddir+'err_%.2f_%.2f_%d.p'%(f,q,st),'rb'))
			tspan /= tspan[-1]
			labf = interp1d(tspan, err_lab, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
			alif = interp1d(tspan, err_ali, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
			latf = interp1d(tspan, err_lat, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
			phf = interp1d(tspan, err_ph, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

			labt = labf(teval)
			alit = alif(teval)
			latt = latf(teval)
			pht = phf(teval)
			# Alignment symmetric wrt rotations by pi
			# NODE sometimes learns to go to 180 degree rotation from true alpha
			# Check in approach learning sin alpha and cos alpha - these two may be the same
			pht -= np.pi*np.round(pht/np.pi)

			# print('strmline %d f %.2f q %.2f st %d'%(count,f,q,st))
			# print(labt_avg)
			labt_avg += labt
			# print(labt_avg)
			# if count == 2:
				# exit()
			alit_avg += alit
			latt_avg += latt
			pht_avg += pht

			count += 1

print(count)
fig = plt.figure(figsize=(4,4),dpi=300)
ax = plt.subplot(211)
plt.plot(teval,latt_avg/count,'o-',label=r'$h$',markersize=3.0,lw=1.0)
plt.plot(teval,pht_avg/count,'s--',label=r'$\alpha$',markersize=3.0,lw=1.0,fillstyle='none')
plt.legend()
plt.ylabel(r'$\langle |h - \hat{h}| \rangle, \langle |\alpha - \hat{\alpha}| \rangle$')
# plt.xlabel(r'$t/t_{strm}$')
ax.text(-0.2,0.95,'a)',transform=ax.transAxes)
plt.xticks([])
plt.xlim(0,1)
plt.ylim(0,)
plt.yticks([0,0.05,0.1])

# ax = plt.subplot(132)
# plt.plot(teval,alit_avg/count,'o-',markersize=3.0,lw=1.0)
# plt.ylabel(r'$\langle |I_{\alpha} - \tilde{\hat{I}}_{\alpha}|/|I| \rangle$')
# plt.xlabel(r'$t/t_{strm}$')
# ax.text(-0.25,0.95,'b)',transform=ax.transAxes)

ax = plt.subplot(212)
plt.plot(teval,labt_avg/count,'o-',markersize=3.0,lw=1.0)
plt.ylabel(r'$\langle |I - \tilde{\hat{I}}|/|I| \rangle$')
plt.xlabel(r'$\mathcal{T}/\mathcal{T}_{max}$')
ax.text(-0.2,0.95,'b)',transform=ax.transAxes)
plt.xlim(0,1)
plt.ylim(0,)
plt.yticks([0,0.02,0.04,0.06])

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(open(ddir2+'avg_err.png','wb'))
plt.show()
# exit()

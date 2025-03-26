import os
import sys
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from itertools import cycle

# flowType = float(sys.argv[1])
# strain_max = float(sys.argv[2])
flowType = 0.0
flowType2 = 1.0

Pelist = np.logspace(0,np.log10(50.0),10)
low = np.array([0.1,0.5])
Pelist = np.append(low,Pelist)
print(Pelist)
avg_win = np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.01,0.01,0.01])
strain_max = np.zeros(len(Pelist)) + 20.0
strain_max[0] = 1.0
strain_max[1] = 5.0
# strain_max = 100.0
Pelist = Pelist[:-2]
avg_win = avg_win[:-2]

# avg_win = 0.01
Dr = 2.0
nqi = 100
qlim = 0.1
# nthph = 300
nthph = 200
npsi = 20
ntraj = 100000
nodes = [48]
ppn = [24]
nnodes = len(nodes)
load = 0

# alpha = np.array([0.0,1.0e-4,1.0e-3,1.0e-2])
# arch = 0
# traj = np.arange(10,13)
# data_size = 95119
# wdir = 'l2c_smooth_sin'
# dh = 5
# ae_model = 0
# t = 11
# movie = 1
# dPCA = 50

alpha = np.array([1.0e-3])
arch = 0
traj = np.arange(9,10)
data_size = 106663
wdir = 'l2c_ups_smooth'
dh = 5
ae_model = 1
t = 11
movie = 1
dPCA = 50

count_threads = 0
node = 0
count = 0

for al in alpha:
	for tr in traj:
		# color = iter(cm.inferno(np.linspace(0, 0.9, len(Pelist))))
		color = cycle(cm.inferno(np.linspace(0, 0.9, len(Pelist))))
		lines = cycle(['-','--',':','-.'])
		ddir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d/extrap/'%(dh,dPCA,wdir,arch,t,al,tr)
		plt.figure(figsize=(8,3),dpi=300)
		ax = plt.subplot(121)
		plt.ylabel(r'$\langle |I - \tilde{\hat{I}}|/|I| \rangle$')
		plt.xlabel(r'$\gamma = \dot{\gamma} t$')
		for Pe in Pelist:
			tspan,err_lab,err_ali,err_lat,err_ph = pickle.load(open(ddir+'err_%.2f_%.2e.p'%(flowType,Pe),'rb'))
			strain = Pe/Dr*tspan
			c = next(color)
			ls = next(lines)
			# c = cycle(color)
			# ls = cycle(lines)
			# print(ls)
			plt.plot(strain,err_lab,c=c,ls=ls,label='Pe %.1f'%Pe)
		# plt.legend(loc='upper right')
		plt.ylim(0)
		plt.xlim(0,20)
		ax.text(-0.1,1.0,'a)',transform=ax.transAxes)

		color = cycle(cm.inferno(np.linspace(0, 0.9, len(Pelist))))
		lines = cycle(['-','--',':','-.'])
		ax = plt.subplot(122)
		# plt.ylabel(r'$\langle |I - \tilde{\hat{I}}|/|I| \rangle$'))
		plt.xlabel(r'$\epsilon = \dot{\epsilon} t$')
		# for Pe in Pelist[:-1]:
		for Pe in Pelist:
			tspan,err_lab,err_ali,err_lat,err_ph = pickle.load(open(ddir+'err_%.2f_%.2e.p'%(flowType2,Pe),'rb'))
			strain = Pe/Dr*tspan
			c = next(color)
			ls = next(lines)
			plt.plot(strain,err_lab,c=c,ls=ls,label='Pe %.1f'%Pe)
		plt.legend(loc='upper right')
		plt.ylim(0)
		plt.xlim(0,20)
		ax.text(-0.1,1.0,'b)',transform=ax.transAxes)

		plt.tight_layout()
		# plt.savefig(open(ddir+'err_vs_Pe_FT_%.2f.png'%flowType,'wb'))
		plt.savefig(open(ddir+'err_vs_Pe.png','wb'))
		plt.close()


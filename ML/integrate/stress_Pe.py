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
Pelist = Pelist[:]
avg_win = avg_win[:]

# avg_win = 0.01
Dr = 2.0
gdot = Pelist*Dr
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
# traj = np.arange(10,13)
alpha = np.array([0.001])
traj = np.array([11])
arch = 0
data_size = 95119
wdir = 'l2c_smooth_sin'
dh = 5
ae_model = 0
t = 11
movie = 1
dPCA = 50

count_threads = 0
node = 0
count = 0

angle = np.pi/4
RQshear = np.array([[np.cos(angle),-np.sin(angle),0.0],[np.sin(angle),np.cos(angle),0.0],[0.0,0.0,1.0]])

for al in alpha:
	for tr in traj:
		# color = iter(cm.inferno(np.linspace(0, 0.9, len(Pelist))))
		color = cycle(cm.inferno(np.linspace(0, 0.9, len(Pelist))))
		lines = cycle(['-','--',':','-.'])
		ddir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d/extrap/'%(dh,dPCA,wdir,arch,t,al,tr)

		stress_xy = np.zeros((len(Pelist),9))
		stress_til_xy = np.zeros((len(Pelist),9))
		stress_hattil_xy = np.zeros((len(Pelist),9))
		stderr_hattil_xy = np.zeros((len(Pelist),9))

		count = 0
		for Pe in Pelist:
			tspan,stress_lab,stress_til_lab,stress_hattil_lab = pickle.load(open(ddir+'stress_%.2f_%.2e.p'%(flowType,Pe),'rb'))
			print('shapes ',stress_lab.shape,stress_til_lab.shape,stress_hattil_lab.shape)
			stress_lab = np.mean(stress_lab[int(0.5*len(stress_lab)):],axis=0)
			stress_til_lab = np.mean(stress_til_lab[int(0.5*len(stress_lab)):],axis=0)
			stderr_hattil_lab = np.std(stress_hattil_lab[int(0.5*len(stress_lab)):],axis=0)
			stress_hattil_lab = np.mean(stress_hattil_lab[int(0.5*len(stress_lab)):],axis=0)
			print('shapes ',stress_lab.shape,stress_til_lab.shape,stress_hattil_lab.shape,stderr_hattil_lab.shape)
			stress_xy[count] = np.reshape(RQshear@np.reshape(stress_lab,(3,3))@np.transpose(RQshear),9)
			stress_til_xy[count] = np.reshape(RQshear@np.reshape(stress_til_lab,(3,3))@np.transpose(RQshear),9)
			stress_hattil_xy[count] = np.reshape(RQshear@np.reshape(stress_hattil_lab,(3,3))@np.transpose(RQshear),9)
			stderr_hattil_xy[count] = np.reshape(RQshear@np.reshape(stderr_hattil_lab,(3,3))@np.transpose(RQshear),9)
			count += 1

		fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(4,4),sharex=True,sharey=False,dpi=300)
		ax = axs[0]
		ax.set_ylabel(r'$\eta_p / \eta_s \phi = \sigma_{xy} / \eta_s \phi \dot{\gamma}$')
		# plt.xlabel(r'$Pe = \dot{\gamma} / D_r$')
		ax.loglog(Pelist,stress_xy[:,1]/gdot,'o',label='Data')
		ax.loglog(Pelist,stress_til_xy[:,1]/gdot,'s',fillstyle='none',label=r'$F(h,\alpha;\theta_F)$')
		# plt.loglog(Pelist,stress_hattil_xy[:,1]/gdot,'x',label=r'$F(\hat{h},\hat{\alpha})$')
		ax.loglog(Pelist,stress_hattil_xy[:,1]/gdot,'x',label='SIMPLE')
		ax.legend(loc='lower center')
		# plt.ylim(0)
		# plt.xlim(0,20)
		ax.text(-0.2,1.0,'a)',transform=ax.transAxes)

		stress_avg = np.zeros((len(Pelist),9))
		stress_til_avg = np.zeros((len(Pelist),9))
		stress_hattil_avg = np.zeros((len(Pelist),9))

		print('Extension')
		count = 0
		for Pe in Pelist:
			tspan,stress_lab,stress_til_lab,stress_hattil_lab = pickle.load(open(ddir+'stress_%.2f_%.2e.p'%(flowType2,Pe),'rb'))
			print('shapes ',stress_lab.shape,stress_til_lab.shape,stress_hattil_lab.shape)
			stress_avg[count] = np.mean(stress_lab[int(0.5*len(stress_lab)):],axis=0)
			stress_til_avg[count] = np.mean(stress_til_lab[int(0.5*len(stress_lab)):],axis=0)
			stress_hattil_avg[count] = np.mean(stress_hattil_lab[int(0.5*len(stress_lab)):],axis=0)
			# print('shapes ',stress_lab.shape,stress_til_lab.shape,stress_hattil_lab.shape)
			# stress_xy[count] = np.reshape(RQshear@np.reshape(stress_lab,(3,3))@np.transpose(RQshear),9)
			# stress_til_xy[count] = np.reshape(RQshear@np.reshape(stress_til_lab,(3,3))@np.transpose(RQshear),9)
			# stress_hattil_xy[count] = np.reshape(RQshear@np.reshape(stress_hattil_lab,(3,3))@np.transpose(RQshear),9)
			count += 1
			# print((stress_avg[:,0] - stress_avg[:,4])/gdot)

		print(stress_hattil_avg[:,0] - stress_hattil_avg[:,4])
		ax = axs[1]
		ax.set_ylabel(r'$\eta_E / \eta_s \phi = (\sigma_{xx} - \sigma_{yy}) / \eta_s \phi \dot{\epsilon}$')
		ax.set_xlabel('Pe'+r'$_{shear} = \dot{\gamma} / D_r$'+', Pe'+r'$_{ext} = \dot{\epsilon} / D_r$')
		ax.loglog(Pelist,(stress_avg[:,0] - stress_avg[:,4])/gdot,'o',label='Data')
		plt.loglog(Pelist,(stress_til_avg[:,0] - stress_til_avg[:,4])/gdot,'s',fillstyle='none')
		ax.loglog(Pelist,(stress_hattil_avg[:,0] - stress_hattil_avg[:,4])/gdot,'x',label='SIMPLE')
		# plt.legend(loc='lower right')
		# plt.ylim(0)
		# plt.xlim(0,20)
		ax.text(-0.2,1.0,'b)',transform=ax.transAxes)

		plt.tight_layout()
		plt.subplots_adjust(hspace=0)
		# plt.savefig(open(ddir+'err_vs_Pe_FT_%.2f.png'%flowType,'wb'))
		plt.savefig(open(ddir+'stress_vs_Pe.png','wb'))
		plt.close()


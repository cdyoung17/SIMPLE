import numpy as np
import scipy
from scipy import integrate
# from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torchinfo import summary
import time

from torchdiffeq import odeint

# Test ODE function
def test (y, t):
    dydt = c1*c2*Efunc(t)[0] - c1*y*Efunc(t)[1] - c3*Efunc(t)[3]
    return dydt

# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,dPCA,alpha):
        super(ODEFunc, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dPCA+6, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, dPCA),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # print(y)
        a = y[0]
        t0 = y[1]
        # print(t)
        # print(t0)
        # print(a.shape)
        E = Efunc(t.detach().numpy() + t0.detach().numpy())
        E = torch.tensor(E)
        E = E.type(torch.FloatTensor)
        aE = torch.cat((a,E),axis=-1)
        # return self.net(aE) - self.alpha*y
        # print(tuple([self.net(aE) - self.alpha*a] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]]))
        return tuple([self.net(aE) - self.alpha*a] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]])
        # return tuple([self.net(aE) - self.alpha*a,0.0])
        # return self.net(aE) - self.alpha*y

# This is the class that contains the NN that estimates the RHS
class SciPyODE(nn.Module):
    def __init__(self,dPCA,alpha):
        super(SciPyODE, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dPCA+6, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, dPCA),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        E = Efunc(t)
        aE = np.concatenate((y,E))
        aE = torch.tensor(aE)
        aE = aE.type(torch.FloatTensor)
        return self.net(aE).detach().numpy() - self.alpha*y

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':

	dPCA = 30
	M = 95181

	# Load the data and put it in the proper form for training
	data_dir = '../../BD_FFoRM/c_rods/training_output/'
	a = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/red_%d_PCA_modes_%d.p'%(dPCA,M),'rb'))
	E = pickle.load(open(data_dir+'stress_L1680.0_R34.0/gradient_snapshots_clearnan_%d.p'%M,'rb'))

	a = torch.tensor(a[:,np.newaxis,:])
	a = a.type(torch.FloatTensor)
	red_sym_ind = np.array([0,1,2,4,5,8])
	E = E[:,red_sym_ind]

	dt = 0.01
	sample_times = np.arange(len(E))*dt
	Efunc = interp1d(sample_times, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

	# Etest = Efunc([0.0,0.01])
	# # print(a[0].shape)
	# print(Etest.shape)
	# aEtest = np.concatenate((a[0:2],Etest),axis=1)
	# print(aEtest.shape)
	# exit()

	c1, c2, c3 = [-0.3, 1.4, -0.5] # co-efficients

	# # Test ODE function
	# def test (y, t):
	#     dydt = c1*c2*Efunc(t)[0] - c1*y*Efunc(t)[1] - c3*Efunc(t)[3]
	#     return dydt

	# tspan = np.arange(0,500)*dt
	# Evolve first PCA coefficient with test function
	# a0 = a[0,0:2]
	# yt = odeint(test, a0, tspan)

	alpha = 0.0
	func = ODEFunc(dPCA,alpha)
	lr_init = 1e-3
	itr_start = 1
	it_per_drop = 10000
	optimizer = optim.Adam(func.parameters(), lr=lr_init) #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=it_per_drop, gamma=0.5)
	end = time.time()
	time_meter = RunningAverageMeter(0.97)
	loss_meter = RunningAverageMeter(0.97)

	# # model_stats = summary(func, [(1,1),(1,1,dPCA)])
	# # model_stats = summary(func, [(1,1),(1,1,dPCA)])
	# model_stats = summary(func, [(1,1),(1,1,tuple([dPCA,1]))])
	# # model_stats = summary(func, [(1,dPCA),(1,1)])
	# model_str = str(model_stats)

	# with open('model_summary.txt', 'w') as f:
	#     # model.summary(print_fn=lambda x: f.write(x + '\n'))
	#     f.write(model_str)

	torch.save(func, 'model.pt')
	torch.save(func.state_dict(),'state_dict.pt')

	# pred_y = odeint(func, a[0], tspan, tfirst=True)
	# sol = scipy.integrate.solve_ivp(func,[0,tpred[-1]],ud[0],t_eval = tpred,max_step=0.01)
	start = 948
	# print(a.shape)
	# print(a[0,0].shape)
	# print(a[start,0])
	# sol = scipy.integrate.solve_ivp(func,[0,tspan[-1]],a[0,0],t_eval=tspan,max_step=0.01)
	nt = 100
	# tspan = sample_times[start:start+nt]
	tspan = sample_times[0:nt]
	tspan = torch.tensor(tspan)
	tspan = tspan.type(torch.FloatTensor)
	# print(tspan)
	# tspan = np.arange(0,500)*dt
	# pred_y = odeint(func, a[start,0], tspan)
	t0 = sample_times[start]
	t0 = torch.tensor(t0)
	t0 = t0.type(torch.FloatTensor)
	pred_y = odeint(func, tuple([a[start,0],t0]), tspan)
	pred_y = pred_y[0].detach().numpy()

	func_sp = SciPyODE(dPCA,alpha)
	func_sp.load_state_dict(torch.load('state_dict.pt'))
	func_sp.eval()

	tspan = sample_times[start:start+nt]
	sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],a[start,0],t_eval=tspan,max_step=0.01)
	pred_y2 = np.transpose(sol['y'])
	print(pred_y2.shape)

	comp1 = 5
	comp2 = 6
	plt.figure(figsize=(4,3),dpi=300)
	plt.plot(tspan, pred_y[:,comp1],'-',label='Test func 0')
	plt.plot(tspan, pred_y[:,comp2],'-',label='Test func 1')
	plt.plot(tspan, pred_y2[:,comp1],'s',fillstyle='none',markersize=2,label='SciPyODE 0')
	plt.plot(tspan, pred_y2[:,comp2],'o',fillstyle='none',markersize=2,label='SciPyODE 1')
	plt.plot(tspan, a[start:start+nt,0,comp1],'s',markersize=1,label='Data 0')
	plt.plot(tspan, a[start:start+nt,0,comp2],'o',markersize=1,label='Data 1')
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel('PCA mode')
	plt.tight_layout()
	plt.show()

	exit()
	#############

	nt = 50000
	dts = 0.001
	nt_data = int(nt*dts/dt)
	# print(nt_data)
	test_t = np.arange(0,nt)*dts
	interp = Efunc(test_t)
	print(interp.shape)

	comp = 6
	plt.figure(figsize=(4,3),dpi=300)
	plt.plot(test_t, interp[:,comp],label='Interpolation')
	plt.plot(sample_times[:nt_data], E[:nt_data,comp],'--',label='Samples')
	plt.legend()
	# plt.plot(test_t, interp[:,1])
	# plt.plot(test_t, interp[:,2])
	# plt.plot(test_t, interp[:,3])
	# plt.plot(test_t, interp[:,4])
	# plt.plot(test_t, interp[:,5])
	# plt.title('Multiple Parameters Test')
	plt.xlabel('Time')
	plt.ylabel('Interpolated E00')
	plt.tight_layout()
	plt.show()
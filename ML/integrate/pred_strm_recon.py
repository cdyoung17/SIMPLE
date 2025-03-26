#! /home/cyoung37/anaconda3/bin/python3
import sys
# sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')
import os
import argparse
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import cm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torchinfo import summary
import parse
import scipy
from scipy.ndimage import uniform_filter1d
from scipy import integrate
from scipy.interpolate import interp1d

# For importing the autoencoder
# sys.path.insert(0, '../')
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--dPCA', type=int, default=50) # Number of PCA modes retained
parser.add_argument('--data_size', type=int, default=95181)  #IC from the simulation
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--batch_time', type=int, default=1)   #Samples a batch covers (this is 10 snaps in a row in data_size)
parser.add_argument('--batch_size', type=int, default=100)   #Number of IC to calc gradient with each iteration
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=40000)       #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)    #Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--arch', type=int, default=0)
parser.add_argument('--traj', type=int, default=0)
parser.add_argument('--FRR', type=float, default=0.0) 
parser.add_argument('--Q2', type=float, default=3.4) 
parser.add_argument('--strm', type=int, default=0)
parser.add_argument('--movie', type=int, default=0)
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

# Determines what solver to use
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    # This is the default
    from torchdiffeq import odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS during training
class ODEFunc(nn.Module):
    def __init__(self,dPCA,alpha):
        super(ODEFunc, self).__init__()
        self.alpha = alpha
        # self.net = nn.Sequential(
        #     nn.Linear(dPCA+6, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200,200),
        #     nn.ReLU(),
        #     nn.Linear(200, dPCA),
        # )
        self.net = nn.Sequential(
            nn.Linear(dPCA+6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dPCA),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # E = Efunc(t.detach().numpy())
        a = y[0]
        t0 = y[1]
        E = Efunc(t.detach().numpy() + t0.detach().numpy())
        E = E[:,np.newaxis,:]
        # print(E.shape)
        # print(a.shape)
        # print(t0.shape)
        E = torch.tensor(E)
        E = E.type(torch.FloatTensor)
        # aE = torch.cat((y,E),axis=-1)
        # return self.net(aE) - self.alpha*y
        aE = torch.cat((a,E),axis=-1)
        return tuple([self.net(aE) - self.alpha*a] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]])

# Use the trained model with SciPy integrators to reduce cost of storing the gradients
class SciPyODE(nn.Module):
    def __init__(self,dPCA,alpha):
        super(SciPyODE, self).__init__()
        self.alpha = alpha
        # self.net = nn.Sequential(
        #     nn.Linear(dPCA+6, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200,200),
        #     nn.ReLU(),
        #     nn.Linear(200, dPCA),
        # )
        self.net = nn.Sequential(
            nn.Linear(dPCA+6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dPCA),
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

# This class is used for updating the gradient
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

###############################################################################
# Functions
###############################################################################
# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(t,true_y,ic_indices):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    s = torch.from_numpy(np.random.choice(ic_indices, args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_t0 = t[s]
    # batch_t = torch.stack([t[s + i] for i in range(args.batch_time)], dim=0)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # return batch_y0, batch_t, batch_y
    return batch_y0, batch_t0, batch_t, batch_y

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

def best_path(path):

    text=open(path+'Trials.txt','r')
    MSE=[]
    for line in text:
        vals=line.split()
        # Check that the line begins with T, meaning it has trial info
        if vals[0][0]=='T':
            # Put MSE data together
            MSE.append(float(vals[1]))
    
    idx=np.argmin(np.asarray(MSE))+1

    return path+'Trial'+str(idx)

def rolling_mean_along_axis(a, W, axis=0):
    # a : Input ndarray
    # W : Window size
    # axis : Axis along which we will apply rolling/sliding mean
    hW = W//2
    L = a.shape[axis]-W+1   
    indexer = [slice(None) for _ in range(a.ndim)]
    indexer[axis] = slice(hW,hW+L)
    # cut_edges = uniform_filter1d(a,W,axis=axis)[tuple(indexer)]
    return uniform_filter1d(a,W,axis=axis)

if __name__ == '__main__':

    batch_time = args.batch_time
    data_size = args.data_size
    dPCA = args.dPCA
    arch = args.arch
    model = args.traj
    dt = args.dt
    alpha = args.alpha
    FRR = args.FRR
    Q2 = args.Q2
    strm = args.strm
    # args.data_size = args.data_size - args.dd*args.tsp

    # Load the data and put it in the proper form for training
    # dPCAin = 30
    # data_dir = '../../BD_FFoRM/c_rods/training_output/'
    # anp = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/red_%d_PCA_modes_%d.p'%(dPCAin,data_size),'rb'))
    # E = pickle.load(open(data_dir+'stress_L1680.0_R34.0/gradient_snapshots_clearnan_%d.p'%data_size,'rb'))

    L = 1680.0;
    R = 34.0;
    volfrac=0.1 # volume fraction of cylinders
    dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
    bkgd = 0.0001 # background scattering in cm^-1
    nbins = 200
    nqi = 100
    qlim = 0.1

    # Load PCA modes
    PCA_path = '../../BD_FFoRM/c_rods/training_output/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/pca_%d_%d.p'%(nbins,nqi,qlim,L,R,30,data_size)
    U = pickle.load(open(PCA_path,'rb'))
    print(U.shape)

    # Qx Qy bounds if not using lists, comment out if using manual Qx,Qy coordinates
    Qxmin = -qlim # qx lower bound in Angstrom^-1
    Qxmax = qlim # qx upper bound in Angstrom^-1
    Qxnum = nqi # number of pixels in x direction
    Qxstep = (Qxmax-Qxmin)/(Qxnum-1)
    # qxlist = np.arange(Qxmin,Qxmax+1e-6,Qxstep)
    qxlist = np.zeros(Qxnum)
    for i in range(Qxnum):
        qxlist[i] = Qxmin + i*Qxstep

    Qymin = -qlim # qy lower bound in Angstrom^-1
    Qymax = qlim # qy upper `bound in Angstrom^-1
    Qynum = nqi # number of pixels in y direction
    Qystep = (Qymax-Qymin)/(Qynum-1)
    qylist = np.arange(Qymin,Qymax+1e-6,Qystep)
    qylist = np.zeros(Qynum)
    for i in range(Qynum):
        qylist[i] = Qymin + i*Qystep

    Qxnum = len(qxlist)
    Qynum = len(qylist)

    # initialize wavevectors where you want to make the calculations
    nQ = Qxnum*Qynum
    Qx = np.zeros(nQ)
    Qy = np.zeros(nQ)
    Qz = np.zeros(nQ)
    xcount = 0
    for Qxi in qxlist:
        ycount = 0
        for Qyi in qylist:
            Qx[xcount + Qxnum*ycount] = Qxi
            Qy[xcount + Qxnum*ycount] = Qyi
            ycount = ycount + 1
        xcount = xcount + 1

    gamma = 0.0

    stress_path = '../../BD_FFoRM/c_rods/training_output/stress_L%.1f_R%.1f/'%(L,R)
    tspan,stress,Gd,E = pickle.load(open(stress_path+'FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    stress = np.reshape(stress,(stress.shape[0],9))
    E = Gd
    E = np.reshape(E,(E.shape[0],9))

    red_sym_ind = np.array([0,1,2,4,5,8])
    E = E[:,red_sym_ind]
    stress = stress[:,red_sym_ind]

    # Create interpolation function for continuous evaluation of E for NODE input
    Efunc = interp1d(tspan, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    I_path = '../../BD_FFoRM/c_rods/training_output/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    P,cum_weight = pickle.load(open(I_path+'FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    I = 1.0e8*np.pi*R**2*L*volfrac*dsld**2*P/cum_weight[:,np.newaxis,np.newaxis] + bkgd
    I = np.reshape(I,(I.shape[0],I.shape[1]*I.shape[2]))

    if np.isnan(I[-1,-1]) or np.isinf(I[-1,-1]):
        # n_win -= 1
        I = I[:-1]
        stress = stress[:-1]
        E = E[:-1]
        tspan = tspan[:-1]

    # Project onto truncated PCA modes
    anp = I @ U[:,:dPCA]
    kernel_size = np.max((1,int(len(anp)/20)))
    anp = rolling_mean_along_axis(anp,kernel_size,0)

    # # Normalize all modes by std of first
    # # if model==0:
    # # a_mean=np.mean(anp[:,0])
    # a_mean = np.mean(anp,axis=0)
    # # a_std=np.std(anp[:,0])
    # a_std=np.std(anp,axis=0)
    # a_std = np.max(a_std)
    # # a = (anp-a_mean)/a_std
    # anp = (anp - a_mean[np.newaxis,:])/a_std
    # # anp = anp/a_std
    # print('mean, std')
    # print(a_mean,a_std)

    # # Normalize all modes individually
    # elif model==1:
    # a_mean = np.mean(anp,axis=0)
    # a_std = np.std(anp,axis=0)
    # a_mean,a_std = pickle.load(open('a_stats_%d.p'%dPCA,'rb'))
    # anp = (anp - a_mean[np.newaxis,:])/a_std[np.newaxis,:]
    a_mean,a_std = pickle.load(open('a_stats_v2_%d.p'%dPCA,'rb'))
    anp = (anp - a_mean[np.newaxis,:])/a_std
    # print('mean, std')
    # print(a_mean,a_std)

    [M,N] = anp.shape

    wdir = 'dPCA%d/arch%d_tau%d_alpha%.1e/model%d'%(dPCA,arch,batch_time,alpha,model)
    # if os.path.exists(wdir) == 0:
    #     os.makedirs(wdir,exist_ok=False)

    os.chdir(wdir)

    # func = torch.load('model.pt')
    # func = torch.load('chp/model.pt')
    # func.eval()

    func_sp = SciPyODE(dPCA,alpha)
    # func_sp.load_state_dict(torch.load('state_dict.pt'))
    func_sp.load_state_dict(torch.load('chp/sd.pt'))
    func_sp.eval()

    nt = len(tspan)
    sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],anp[0],t_eval=tspan,max_step=0.01)
    pred_y = np.transpose(sol['y'])

    start = 0
    nrows = 2
    ncols = 3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,6),sharex=True,)

    count = 0
    for ax in axs.ravel():
        # print(ax)
        ax.plot(tspan,anp[:,count],'o',markersize=1,label='Data')
        ax.plot(tspan,pred_y[:,count],'s',markersize=1,label='Prediction')
        # ax.set_ylabel(r'$a_{%d}$'%(count+1))
        ax.set_title('Mode %d'%(count+1))
        # ax.set_ylabel(r'$t$')
        # ax.set_ylabel(r'$q_y$')
        # cb = fig.colorbar(cax,ax=ax)
        # cb.ax.set_title(r'$a(q)$')

        count = count + 1

    fig.supxlabel(r'$t$')
    plt.tight_layout()
    plt.savefig(open('pred_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    plt.show()

    # pred_y = pred_y*a_std[np.newaxis,:] + a_mean[np.newaxis,:]
    # anp = anp*a_std[np.newaxis,:] + a_mean[np.newaxis,:]
    pred_y = pred_y*a_std + a_mean[np.newaxis,:]
    anp = anp*a_std + a_mean[np.newaxis,:]
    Ihattil = pred_y @ np.transpose(U[:,:dPCA])
    Itil = anp @ np.transpose(U[:,:dPCA])
    I = np.reshape(I,(I.shape[0],nqi,nqi))
    Itil = np.reshape(Itil,(Itil.shape[0],nqi,nqi))
    Ihattil = np.reshape(Ihattil,(Ihattil.shape[0],nqi,nqi))
    # vmin = np.min(Itil)
    # vmax = np.max(Itil)
    # errmin = np.min(np.sqrt((I - Ihattil)**2))
    # errmax = np.max(np.sqrt((I - Ihattil)**2))
    errmin = 1e-4
    errmax = 5e2

    if args.movie == 1:
        tmov = np.linspace(0,30,len(Itil))
        # mymap=matplotlib.cm.get_cmap('inferno')
        mymap=matplotlib.cm.get_cmap('jet')

        fig = plt.figure(figsize=(10.5,3),dpi=300)
        ax3 = plt.subplot(131)
        cax3 = ax3.pcolormesh(qxlist,qylist,I[0],norm=colors.LogNorm(),cmap=mymap,shading='gouraud')
        ax3.set_aspect('equal')
        ax3.set_xlabel(r'$q_x$')
        ax3.set_ylabel(r'$q_y$')
        cb3 = fig.colorbar(cax3,ax=ax3)
        cb3.ax.set_title(r'$I(q)$')

        ax4 = plt.subplot(132)
        # cax4 = ax4.pcolormesh(qxlist,qylist,Itil[0],norm=colors.LogNorm(),cmap=mymap,shading='gouraud')
        cax4 = ax4.pcolormesh(qxlist,qylist,Ihattil[0],norm=colors.SymLogNorm(linthresh=1e-5, linscale=0.03, base=10),cmap=mymap,shading='gouraud')
        ax4.set_aspect('equal')
        ax4.set_xlabel(r'$q_x$')
        # ax4.set_ylabel(r'$q_y$')
        cb4 = fig.colorbar(cax4,ax=ax4)
        cb4.ax.set_title(r'$\hat{\tilde{I}}(q)$')

        ax2 = plt.subplot(133)
        # cax2 = ax2.pcolormesh(qxlist,qylist,Itil[0],norm=colors.LogNorm(),cmap=mymap,shading='gouraud')
        cax2 = ax2.pcolormesh(qxlist,qylist,np.sqrt((I[0] - Ihattil[0])**2),norm=colors.LogNorm(vmin=errmin,vmax=errmax),cmap=mymap,shading='gouraud')
        ax2.set_aspect('equal')
        ax2.set_xlabel(r'$q_x$')
        # ax2.set_ylabel(r'$q_y$')
        cb2 = fig.colorbar(cax2,ax=ax2)
        cb2.ax.set_title(r'$|I(q) - \hat{\tilde{I}}(q)|$')

        plt.tight_layout()

        def animate(i):
           # ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           cax3.set_array(I[i])
           cax4.set_array(Ihattil[i])
           cax2.set_array(np.sqrt((I[i] - Ihattil[i])**2))
           ax3.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           ax2.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           ax4.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))

        anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(tmov))
        anim.save('FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))
        # plt.show()

    aE = np.concatenate((pred_y,E),axis=1)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(aE.shape[1])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128,activation='relu'))
    model.add(keras.layers.Dense(units=128,activation='relu'))
    model.add(keras.layers.Dense(stress.shape[1]))

    model.compile(loss='mse', optimizer='adam')

    # Save the architecture summary for quick viewing
    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    archs = 2
    regs = 0.0001
    models = 0
    stress_path = '../../../../stress/dPCA%d/arch%d_reg%.6f/model%d/'%(dPCA,archs,regs,models)
    if os.path.exists(stress_path+'model.h5'):
        model.load_weights(stress_path+'model.h5')
    else:
        model.load_weights(stress_path+'model_chpt.h5')

    stress_pred = model.predict(aE)

    fig = plt.figure(figsize=(8,6),dpi=200)

    # ax = plt.subplot(111)
    ax = plt.subplot(221)
    ax.plot(tspan,stress[:,0],'s',markersize=1,label='Data')
    ax.plot(tspan,stress_pred[:,0],'o',markersize=1,label='Prediction')
    # ax.plot(t,E[:,0],label='E_{00}')
    plt.legend()
    # plt.title('MSE %.3e'%mse)
    ax.set_ylabel(r'$\sigma_{xx}$')

    ax = plt.subplot(222)
    ax.plot(tspan,stress[:,1],'s',markersize=1)
    ax.plot(tspan,stress_pred[:,1],'o',markersize=1)
    ax.set_ylabel(r'$\sigma_{xy}$')

    ax = plt.subplot(223)
    ax.plot(tspan,stress[:,3],'s',markersize=1)
    ax.plot(tspan,stress_pred[:,3],'o',markersize=1)
    ax.set_ylabel(r'$\sigma_{yy}$')
    ax.set_xlabel(r'$t$')

    ax = plt.subplot(224)
    ax.plot(tspan,stress[:,5],'s',markersize=1)
    ax.plot(tspan,stress_pred[:,5],'o',markersize=1)
    ax.set_ylabel(r'$\sigma_{zz}$')
    ax.set_xlabel(r'$t$')

    plt.tight_layout()

    plt.savefig(open('str_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))

    # fig = plt.figure(figsize=(12,3),dpi=200)

    # # ax = plt.subplot(111)
    # ax = plt.subplot(131)
    # ax.plot(tspan,E[:,0],'s',markersize=1)
    # # plt.legend()
    # ax.set_ylabel(r'$E_{xx}$')
    # ax.set_xlabel(r'$t$')

    # ax = plt.subplot(132)
    # ax.plot(tspan,E[:,1],'s',markersize=1)
    # ax.set_ylabel(r'$E_{xy}$')
    # ax.set_xlabel(r'$t$')

    # ax = plt.subplot(133)
    # ax.plot(tspan,E[:,3],'s',markersize=1)
    # ax.set_ylabel(r'$E_{yy}$')
    # ax.set_xlabel(r'$t$')
    # plt.tight_layout()

    # plt.show()


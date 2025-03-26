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
import torch as T
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torchinfo import summary
import parse

import scipy
from scipy.ndimage import uniform_filter1d
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import ndimage, misc

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
parser.add_argument('--dh', type=int, default=5) # Autoencoder latent dimension
parser.add_argument('--data_size', type=int, default=95119)  #IC from the simulation
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--batch_time', type=int, default=10)   #Samples a batch covers (this is 10 snaps in a row in data_size)
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
parser.add_argument('--FRR', type=float, default=-0.7) 
parser.add_argument('--Q2', type=float, default=10.0) 
parser.add_argument('--strm', type=int, default=4)
parser.add_argument('--movie', type=int, default=0)
parser.add_argument('--dir', type=str, default='') 
parser.add_argument('--ae_model', type=int, default=0)
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

class SciPyODE3(nn.Module):
    def __init__(self,dh,alpha):
        super(SciPyODE3, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.netornt = nn.Sequential(
            nn.Linear(dh+10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for m in self.netornt.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        h = y[:-1]
        ornt = y[-1]

        # E1 = aTEafunc(t)
        # E2 = QTEQfunc(t)
        E = Efunc(t)
        Q = Qfunc(t)
        thetaQ = np.arctan2(Q[1],Q[0])
        phaseQ = -ornt/2.0 + thetaQ
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,0.0]])
        E = np.reshape(E,(3,3))
        Q = np.reshape(Q,(3,3))
        E1 = np.transpose(RQ)@E@RQ
        E2 = Q@E@np.transpose(Q)
        E1 = np.reshape(E1,9)
        E2 = np.reshape(E2,9)

        # yE1 = np.concatenate((y,E1))
        yE1 = np.concatenate((y,E1))
        yE1 = torch.tensor(yE1)
        yE1 = yE1.type(torch.FloatTensor)
        yE2 = np.concatenate((y,E2))
        yE2 = torch.tensor(yE2)
        yE2 = yE2.type(torch.FloatTensor)
        # exit()

        hRHS = self.net(yE1).detach().numpy() - self.alpha*h
        thetaRHS = self.netornt(yE2).detach().numpy() - self.alpha*ornt
        # print(hRHS.shape,thetaRHS.shape)
        RHS = np.append(hRHS,thetaRHS,axis=-1)

        return RHS

class SciPyODE2(nn.Module):
    def __init__(self,dh,alpha):
        super(SciPyODE2, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+9, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.netornt = nn.Sequential(
            nn.Linear(dh+9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for m in self.netornt.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        h = y[:-1]
        ornt = y[-1]
        g = gfunc(t)
        hg = np.concatenate((h,g))
        hg = torch.tensor(hg)
        # print(hg.shape)
        # print(ornt.shape)
        hg = hg.type(torch.FloatTensor)

        hRHS = self.net(hg).detach().numpy() - self.alpha*h
        thetaRHS = self.netornt(hg).detach().numpy() - self.alpha*ornt
        # RHS = torch.cat((hRHS,thetaRHS),axis=-1)
        RHS = np.append(hRHS,thetaRHS,axis=-1)
        return RHS

# Use the trained model with SciPy integrators to reduce cost of storing the gradients
class SciPyODE(nn.Module):
    def __init__(self,dh,alpha):
        super(SciPyODE, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh+1),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(dh+6, 64),
        #     nn.ReLU(),
        #     nn.Linear(64,64),
        #     nn.ReLU(),
        #     nn.Linear(64, dh),
        # )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # g = QTEQfunc(t)
        g = gfunc(t)
        hg = np.concatenate((y,g))
        hg = torch.tensor(hg)
        hg = hg.type(torch.FloatTensor)
        return self.net(hg).detach().numpy() - self.alpha*y

class autoencoder(nn.Module):
    def __init__(self, ambient_dim=30, code_dim=20, nlin=4, filepath='testae'):
        super(autoencoder, self).__init__()
        
        self.ambient_dim = ambient_dim
        self.code_dim = code_dim
        self.nlin = nlin
        
        self.encoder = nn.Sequential(
            nn.Linear(self.ambient_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.code_dim))

        self.linears = nn.ModuleList([nn.Linear(self.code_dim, self.code_dim) for i in range(self.nlin)])
        
        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.ambient_dim))
        
    def forward(self,x):
        code = self.encoder(x)
        for l in enumerate(self.linears):
            code = l[1](code)
        xhat = self.decoder(code)
        return xhat
    
    def encode(self, x):
        code = self.encoder(x)
        for l in enumerate(self.linears):
            # print(l[1])
            code = l[1](code)
        return code
    
    def decode(self, code):
        xhat = self.decoder(code)
        return xhat

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    

###############################################################################
# Functions
###############################################################################

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
    dh = args.dh
    arch = args.arch
    modl = args.traj
    dt = args.dt
    alpha = args.alpha
    FRR = args.FRR
    Q2 = args.Q2
    strm = args.strm
    ae_model = args.ae_model

    wdir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d'%(dh,dPCA,args.dir,arch,batch_time,alpha,modl)

    L = 1680.0;
    R = 34.0;
    volfrac=0.1 # volume fraction of cylinders
    dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
    bkgd = 0.0001 # background scattering in cm^-1
    nbins = 200
    nqi = 100
    qlim = 0.1

    # Load PCA modes
    # PCA_path = '../../BD_FFoRM/c_rods/training_output/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/pca_%d_%d.p'%(nbins,nqi,qlim,L,R,dPCA,data_size)
    # U = pickle.load(open(PCA_path,'rb'))

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

    Qmag = np.sqrt(Qx**2 + Qy**2)
    ind = np.nonzero(Qmag < 0.10)
    print('fill indices',ind)
    gamma = 0.0

    # stress_path = '../../BD_FFoRM/c_rods/training_output/stress_L%.1f_R%.1f/'%(L,R)
    # tspan,stress,Gd,E = pickle.load(open(stress_path+'FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    # stress = np.reshape(stress,(stress.shape[0],9))
    # E = Gd
    # E = np.reshape(E,(E.shape[0],9))
    # Efull = E
    # Gd = np.reshape(Gd,(Gd.shape[0],9))

    # red_sym_ind = np.array([0,1,2,4,5,8])
    # E = E[:,red_sym_ind]
    # Gd = Gd[:,red_sym_ind]
    # stress = stress[:,red_sym_ind]


    I_path = '../../BD_FFoRM/c_rods/training_output/smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    I_path2 = '../../BD_FFoRM/c_rods/training_output/smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    # I_path = '../../BD_FFoRM/c_rods/training_output/l2c_smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    # I_path2 = '../../BD_FFoRM/c_rods/training_output/l2c_smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    IQa,Q,phase_t,Gd,E = pickle.load(open(I_path+'IQphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    # Q,phase_t = pickle.load(open(I_path+'phaseQ_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    tspan = np.arange(len(IQa))*dt

    PCA_path = I_path2+'Qphase_pca_%d.p'%(data_size)
    U,S = pickle.load(open(PCA_path,'rb'))
    print(U.shape)

    # Create interpolation function for continuous evaluation of E for NODE input
    gfunc = interp1d(tspan, Gd, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Efunc = interp1d(tspan, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    # phase_mean,phase_std = pickle.load(open('phase_stats.p','rb'))
    # phase_t = (phase_t - phase_mean)/phase_std
    Q = np.reshape(Q,(Q.shape[0],3,3))
    Gd = np.reshape(Gd,(Gd.shape[0],3,3))
    E = np.reshape(E,(E.shape[0],3,3))
    QT = np.transpose(Q,axes=[0,2,1])
    # QTEQ = QT@E@Q
    QTEQ = np.zeros(E.shape)
    aTEa = np.zeros(E.shape)
    thetaQ = np.zeros(len(Q))
    for i in range(E.shape[0]):
        # QTEQ[i] = QT[i]@E[i]@Q[i]
        QTEQ[i] = Q[i]@E[i]@QT[i]
        thetaQ[i] = np.arctan2(Q[i,0,1],Q[i,0,0])
        phaseQ = -phase_t[i]/2.0 + thetaQ[i]
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,0.0]])
        # print(phaseQ.shape,RQ.shape)
        # exit()
        aTEa[i] = np.transpose(RQ)@E[i]@RQ
        # aTEa[i] = np.transpose(RQ)@Gd[i]@RQ

    Q = np.reshape(Q,(Q.shape[0],9))
    E = np.reshape(E,(E.shape[0],9))
    QTEQ = np.reshape(QTEQ,(QTEQ.shape[0],9))
    aTEa = np.reshape(aTEa,(aTEa.shape[0],9))
    Gd = np.reshape(Gd,(Gd.shape[0],9))
    QTEQfunc = interp1d(tspan, QTEQ, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Efunc = interp1d(tspan, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Qfunc = interp1d(tspan, Q, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    aTEafunc = interp1d(tspan, aTEa, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    start_snap = 0
    # start_snap = len(IQa) - 30
    # n_win -= start_snap
    tspan = tspan[start_snap:]
    IQa = IQa[start_snap:]
    phase_t = phase_t[start_snap:]

    n_win = len(IQa)
    Ifill = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Ifill[i,ind] = IQa[i]
    Ifill = np.reshape(Ifill,(n_win,nqi,nqi))

    # cutoff = int(0.1*len(I))
    # I = I[cutoff:-cutoff]
    # stress = stress[cutoff:-cutoff]
    # E = E[cutoff:-cutoff]
    # Gd = Gd[cutoff:-cutoff]
    # tspan = tspan[cutoff:-cutoff]

    # Project onto truncated PCA modes
    a = IQa @ U[:,:dPCA]
    print(a.shape)
    print(dPCA)
    # kernel_size = np.max((1,int(len(a)/20)))
    # a = rolling_mean_along_axis(a,kernel_size,0)
    # phase_t = rolling_mean_along_axis(phase_t,kernel_size,0)

    # a_mean,a_std = pickle.load(open('a_stats_v2_%d.p'%dPCA,'rb'))
    # anp = (anp - a_mean[np.newaxis,:])/a_std
    # [M,N] = anp.shape

    # DNN AE
    # ae_dir = 'Qphase_ma'
    ae_dir = 'Ismooth'
    model = autoencoder(ambient_dim=dPCA).to(device)
    ae_path = '../autoencoder/dPCA%d/%s/rmodel%d/'%(dPCA,ae_dir,ae_model)
    if(os.path.exists('IRMAE_AE.pt')) == 1:
        model.load_state_dict(T.load(ae_path+'IRMAE_AE.pt'))
    else:
        model.load_state_dict(T.load(ae_path+'IRMAE_AE_chp.pt'))
    a_mean,a_std = pickle.load(open(ae_path+'a_stats_v2_%d.p'%dPCA,'rb'))
    a = (a - a_mean[np.newaxis,:])/a_std

    aT = T.tensor(a, dtype=T.float).to(device)
    b = model.encode(aT)
    b = b.detach().numpy()
    print(b.shape)
    u,s,v = pickle.load(open(ae_path+'code_svd.p','rb'))
    h = b @ u[:,:dh]
    print('ROM shape',h.shape)
    print('phase shape',phase_t.shape)
    ho = np.append(h,phase_t[:,np.newaxis],axis=1)
    print('State shape',ho.shape)

    os.chdir(wdir)

    func_sp = SciPyODE3(dh,alpha)
    # func_sp.load_state_dict(torch.load('state_dict.pt'))
    func_sp.load_state_dict(torch.load('chp/sd.pt'))
    func_sp.eval()

    nt = len(tspan)
    sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],ho[0],t_eval=tspan,max_step=dt)
    pred_y = np.transpose(sol['y'])
    pred_phase = pred_y[:,-1]
    pred_y = pred_y[:,:-1]
    print('predicted shapes',pred_y.shape,pred_phase.shape)

    bhat = pred_y @ np.transpose(u[:,:dh])
    bhat = T.tensor(bhat, dtype=T.float).to(device)
    ahat = model.decode(bhat)
    ahat = ahat.detach().numpy()
    ahat = ahat*a_std + a_mean[np.newaxis,:]
    a = a*a_std + a_mean[np.newaxis,:]
    Ihattil = ahat @ np.transpose(U[:,:dPCA])
    # I = np.reshape(I,(I.shape[0],nqi,nqi))
    Ihattilf = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Ihattilf[i,ind] = Ihattil[i]
    Ihattilf = np.reshape(Ihattilf,(Ihattilf.shape[0],nqi,nqi))

    nrows = 5
    ncols = 5
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,8),sharex=True,)

    count = 0
    for ax in axs.ravel():
        # print(ax)
        ax.plot(tspan,a[:,count],'o',markersize=1)
        ax.plot(tspan,ahat[:,count],'s',markersize=1)
        count = count + 1

    # fig.supxlabel(r'$t$')
    plt.tight_layout()
    plt.savefig(open('pca_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # exit()

    Irot = np.zeros(Ifill.shape)
    for i in range(n_win):   
        Irot[i] = ndimage.rotate(Ifill[i],phase_t[i]/2.0*180.0/np.pi - thetaQ[i]*180.0/np.pi,reshape=False)

    Ihtr = np.zeros(Ihattilf.shape)
    for i in range(n_win):   
        Ihtr[i] = ndimage.rotate(Ihattilf[i],pred_phase[i]/2.0*180.0/np.pi - thetaQ[i]*180.0/np.pi,reshape=False)

    Qmag = np.reshape(Qmag,(nqi,nqi))
    Irot = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, Irot)
    Ihtr = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, Ihtr)

    fig = plt.figure(figsize=(12,6),dpi=300)
    ax = plt.subplot(231)
    plt.plot(tspan,h[:,0],'o-',label='Data')
    plt.plot(tspan,pred_y[:,0],'s--',label='Prediction')
    plt.legend()
    plt.ylabel(r'$h_0$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(232)
    plt.plot(tspan,h[:,1],'o-')
    plt.plot(tspan,pred_y[:,1],'s--')
    plt.ylabel(r'$h_1$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(233)
    plt.plot(tspan,h[:,2],'o-')
    plt.plot(tspan,pred_y[:,2],'s--')
    plt.ylabel(r'$h_2$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(234)
    plt.plot(tspan,h[:,3],'o-')
    plt.plot(tspan,pred_y[:,3],'s--')
    plt.ylabel(r'$h_3$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(235)
    plt.plot(tspan,h[:,4],'o-')
    plt.plot(tspan,pred_y[:,4],'s--')
    plt.ylabel(r'$h_4$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(236)
    plt.plot(tspan,phase_t,'o-')
    plt.plot(tspan,pred_phase,'s--')
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'$t$')

    plt.tight_layout()
    plt.savefig(open('Qphase_enc_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # plt.show()

    # vmin = np.min(Itil)
    # vmax = np.max(Itil)
    # errmin = np.min(np.sqrt((I - Ihattil)**2))
    # errmax = np.max(np.sqrt((I - Ihattil)**2))
    errmin = 1e-4
    errmax = 10.0
    Iq_norm = np.mean(np.sqrt(IQa**2))
    # print(IQa[-1,-1])
    # vmin = np.min(Ihattil)
    # if vmin < 0:
    #     vmin = np.min(I)
    # print(vmin)
    vmin = 0.01

    if args.movie == 1:
        tmov = np.linspace(0,30,len(IQa))
        # mymap=matplotlib.cm.get_cmap('inferno')
        mymap=matplotlib.cm.get_cmap('jet')

        fig = plt.figure(figsize=(10.5,6),dpi=200)

        ax1 = plt.subplot(231)
        cax1 = ax1.pcolormesh(qxlist,qylist,Ifill[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        ax1.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        ax1.set_ylabel(r'$q_y$')
        # ax1.set_title('Data phase-aligned')

        ax2 = plt.subplot(232)
        cax2 = ax2.pcolormesh(qxlist,qylist,Ihattilf[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        ax2.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        # ax.set_ylabel(r'$q_y$')
        ax2.set_title('Model phase-aligned')

        ax3 = plt.subplot(233)
        cax3 = ax3.pcolormesh(qxlist,qylist,np.sqrt((Ifill[0] - Ihattilf[0])**2)/Iq_norm,norm=colors.LogNorm(vmin=errmin,vmax=errmax),cmap=mymap,shading='gouraud')
        ax3.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        # ax.set_ylabel(r'$q_y$')
        cb3 = fig.colorbar(cax3,ax=ax3)
        # cb3.ax.set_title(r'$I_R(q)$')
        cb3.ax.set_title(r'$|I_R(q) - \hat{\tilde{I}}_R(q)|$')
        ax3.set_title('Error phase-aligned')

        ax4 = plt.subplot(234)
        cax4 = ax4.pcolormesh(qxlist,qylist,Irot[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        ax4.set_aspect('equal')
        ax4.set_xlabel(r'$q_x$')
        ax4.set_ylabel(r'$q_y$')
        ax4.set_title('Data lab frame')

        ax5 = plt.subplot(235)
        cax5 = ax5.pcolormesh(qxlist,qylist,Ihtr[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        ax5.set_aspect('equal')
        ax5.set_xlabel(r'$q_x$')
        # ax.set_ylabel(r'$q_y$')
        ax5.set_title('Model lab frame')

        ax6 = plt.subplot(236)
        cax6 = ax6.pcolormesh(qxlist,qylist,np.sqrt((Irot[0] - Ihtr[0])**2)/Iq_norm,norm=colors.LogNorm(vmin=errmin,vmax=errmax),cmap=mymap,shading='gouraud')
        ax6.set_aspect('equal')
        ax.set_xlabel(r'$q_x$')
        # ax.set_ylabel(r'$q_y$')
        cb6 = fig.colorbar(cax6,ax=ax6)
        # cb3.ax.set_title(r'$I_R(q)$')
        cb6.ax.set_title(r'$|I(q) - \hat{\tilde{I}}(q)|/\langle |I(q)| \rangle$')
        ax6.set_title('Error lab frame')

        plt.tight_layout()

        def animate(i):
           # ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           cax1.set_array(Ifill[i])
           cax2.set_array(Ihattilf[i])
           cax3.set_array(np.sqrt((Ifill[i] - Ihattilf[i])**2)/Iq_norm)
           cax4.set_array(Irot[i])
           cax5.set_array(Ihtr[i])
           cax6.set_array(np.sqrt((Irot[i] - Ihtr[i])**2)/Iq_norm)
           # cax5.set_array(Irot[i])
           # cax6.set_array(Ihtr[i])
           # cax7.set_array(np.sqrt((Irot[i] - Ihtr[i])**2))
           ax1.set_title('Data phase-aligned '+r'$\tilde{t} = %.2f$'%(i*0.01))
           # ax2.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           # ax3.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           # ax5.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           # ax6.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))
           # ax7.set_title(r'$\tilde{t} = %.2f$'%(i*0.01))

        # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(tmov))
        nframes = 100
        frames = np.linspace(0,len(IQa),num=nframes,endpoint=False,dtype=int)
        print(frames)
        anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=frames)
        anim.save('Qphase_FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))
        # plt.show()


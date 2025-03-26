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
from tensorflow.keras import regularizers

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
parser.add_argument('--dir', type=str, default='l2c_smooth_sin') 
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

class ODEsin(nn.Module):
    def __init__(self,dh,alpha):
        super(ODEsin, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+11, 128),
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
            nn.Linear(dh+11, 32),
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
        sina = np.sin(ornt)
        cosa = np.cos(ornt)
        # print(h,ornt,sina,cosa)
        # exit()
        # print(h.shape,ornt.shape,sina.shape,cosa.shape)
        y = np.concatenate((h,sina[np.newaxis],cosa[np.newaxis]))
        # y = np.concatenate((h,sina),axis=-1)
        # y = np.concatenate((y,cosa),axis=-1)

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

    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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

    # I_path = '../../BD_FFoRM/c_rods/training_output/%s_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(args.dir,nbins,nqi,qlim,L,R)
    I_path = '../../BD_FFoRM/c_rods/training_output/%s_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%('l2c_ups_smooth',nbins,nqi,qlim,L,R)
    IQa,Q,phase_t,Gd,E,stress = pickle.load(open(I_path+'IQphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    # Q,phase_t = pickle.load(open(I_path+'phaseQ_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    tspan = np.arange(len(IQa))*dt

    # PCA_path = I_path+'Qphase_pca_%d.p'%(data_size)
    PCA_path = '../../BD_FFoRM/c_rods/training_output/%s_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/Qphase_pca_%d.p'%('l2c_smooth',nbins,nqi,qlim,L,R,data_size)
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

    # Compute flow type parameter and flow strength
    Gd_tr = np.transpose(Gd,axes=[0,2,1])
    vort = 0.5*(Gd - Gd_tr)
    vmag = np.einsum('ijk,ijk->i',vort,vort)**0.5
    Emag = np.einsum('ijk,ijk->i',E,E)**0.5
    flow_type = (Emag - vmag)/(Emag + vmag)
    flow_type[flow_type == np.inf] = 0
    GamMag = np.einsum('ijk,ijk->i',Gd,Gd)**0.5
    G = GamMag/np.sqrt(1 + flow_type**2)

    # Flatten deformation tensors for input to NNs
    Q = np.reshape(Q,(Q.shape[0],9))
    E33 = E
    E = np.reshape(E,(E.shape[0],9))
    QTEQ = np.reshape(QTEQ,(QTEQ.shape[0],9))
    aTEa = np.reshape(aTEa,(aTEa.shape[0],9))
    Gd = np.reshape(Gd,(Gd.shape[0],9))
    QTEQfunc = interp1d(tspan, QTEQ, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Efunc = interp1d(tspan, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Qfunc = interp1d(tspan, Q, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    aTEafunc = interp1d(tspan, aTEa, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    start_snap = 50
    # start_snap = len(IQa) - 30
    # n_win -= start_snap
    tspan = tspan[start_snap:]
    IQa = IQa[start_snap:]
    phase_t = phase_t[start_snap:]
    aTEa = aTEa[start_snap:]
    stress = stress[start_snap:]
    flow_type = flow_type[start_snap:]
    E = E[start_snap:]
    G = G[start_snap:]
    QTEQ = QTEQ[start_snap:]
    thetaQ = thetaQ[start_snap:]

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
    # ae_dir = 'l2c'
    model = autoencoder(ambient_dim=dPCA).to(device)
    ae_path = '../autoencoder/dPCA%d/%s/rmodel%d/'%(dPCA,'l2c_smooth',ae_model)
    if(os.path.exists('IRMAE_AE.pt')) == 1:
        model.load_state_dict(T.load(ae_path+'IRMAE_AE.pt'))
    else:
        model.load_state_dict(T.load(ae_path+'IRMAE_AE_chp.pt'))
    a_mean,a_std = pickle.load(open(ae_path+'a_stats_v2_%d.p'%dPCA,'rb'))
    a = (a - a_mean[np.newaxis,:])/a_std

    archstr = 0
    reg = 0.0001
    models = 0
    # Load stress model
    modelstr = keras.Sequential()
    modelstr.add(keras.layers.Input(shape=(dh+9)))
    modelstr.add(keras.layers.Flatten())
    modelstr.add(keras.layers.Dense(units=128,activation='relu',kernel_regularizer=regularizers.l1(reg)))
    modelstr.add(keras.layers.Dense(units=128,activation='relu',kernel_regularizer=regularizers.l1(reg)))
    modelstr.add(keras.layers.Dense(9))

    modelstr.compile(loss='mse', optimizer='adam')

    modelstr.load_weights('../stress/dPCA%d/rdh%d/arch%d_reg%.6f/model%d/model_chpt.h5'%(dPCA,dh,archstr,reg,models))

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

    # Prediciton without time evolution, PCA+AE only
    btil = h @ np.transpose(u[:,:dh])
    btil = T.tensor(btil, dtype=T.float).to(device)
    atil = model.decode(btil)
    atil = atil.detach().numpy()
    atil = atil*a_std + a_mean[np.newaxis,:]
    Itil = atil @ np.transpose(U[:,:dPCA])
    Itilf = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Itilf[i,ind] = Itil[i]
    Itilf = np.reshape(Itilf,(Itilf.shape[0],nqi,nqi))

    # Note - E sampled at regular intervals on output in IQ_phase, so aTEa is here as well
    # In general be careful when input E is form OpenFOAM versus post-processing interpolation to time step

    hE = np.concatenate((h,aTEa),axis=1)
    stress_til = modelstr(hE)

    os.chdir(wdir)
    if os.path.exists('interp') == 0:
        os.makedirs('interp')

    plt.figure(figsize=(8,3))

    ax = plt.subplot(121)
    # plt.plot(t,flow_type)
    plt.plot(tspan,flow_type)
    plt.xlabel('Time')
    plt.ylabel('Flow type parameter '+r'$\Lambda$')

    ax = plt.subplot(122)
    # plt.plot(t,G)
    plt.plot(tspan,G)
    plt.xlabel('Time')
    plt.ylabel('Flow strength G')

    plt.tight_layout()
    plt.savefig(open('interp/LG_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))

    fig = plt.figure(figsize=(8,6),dpi=300)
    ax = plt.subplot(221)
    plt.plot(tspan,E[:,0],'-',label='Lab')
    plt.plot(tspan,aTEa[:,0],'--',label='Phase-aligned')
    plt.plot(tspan,QTEQ[:,0],':',label='Co-rotating')
    plt.legend()
    plt.ylabel(r'$E_{xx}$')
    # plt.xlabel(r'$t$')

    ax = plt.subplot(222)
    plt.plot(tspan,E[:,1],'-',label='Lab')
    plt.plot(tspan,aTEa[:,1],'--',label='Phase-aligned')
    plt.plot(tspan,QTEQ[:,1],':',label='Co-rotating')
    # plt.legend()
    plt.ylabel(r'$E_{xy}$')

    ax = plt.subplot(223)
    plt.plot(tspan,E[:,3],'-',label='Lab')
    plt.plot(tspan,aTEa[:,3],'--',label='Phase-aligned')
    plt.plot(tspan,QTEQ[:,3],':',label='Co-rotating')
    # plt.legend()
    plt.ylabel(r'$E_{yx}$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(224)
    plt.plot(tspan,E[:,4],'-',label='Lab')
    plt.plot(tspan,aTEa[:,4],'--',label='Phase-aligned')
    plt.plot(tspan,QTEQ[:,4],':',label='Co-rotating')
    # plt.legend()
    plt.ylabel(r'$E_{yy}$')
    plt.xlabel(r'$t$')

    plt.tight_layout()
    plt.savefig(open('interp/flow_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))

    # func_sp = SciPyODE3(dh,alpha)
    func_sp = ODEsin(dh,alpha)
    # func_sp.load_state_dict(torch.load('state_dict.pt'))
    func_sp.load_state_dict(torch.load('chp/sd.pt'))
    func_sp.eval()

    nt = len(tspan)
    sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],ho[0],t_eval=tspan,max_step=dt)
    pred_sol = np.transpose(sol['y'])
    pred_phase = pred_sol[:,-1]
    pred_y = pred_sol[:,:-1]
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

    # For time evolution predictions of stress, need to use predicted alpha for rotation of input E
    aTEa_pred = np.zeros(E33.shape)
    for i in range(E.shape[0]):
        phaseQ = -pred_phase[i]/2.0 + thetaQ[i]
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,1.0]])
        aTEa_pred[i] = np.transpose(RQ)@E33[i]@RQ

    aTEa_pred = np.reshape(aTEa,(aTEa.shape[0],9))
    hEpred = np.concatenate((pred_y,aTEa_pred),axis=1)
    stress_hattil = modelstr(hEpred)

    # Rotate back to lab frame for reporting predictions
    stress_lab = np.zeros((n_win,9))
    stress_til_lab = np.zeros((n_win,9))
    stress_hattil_lab = np.zeros((n_win,9))
    for i in range(n_win):
        phaseQ = -phase_t[i]/2.0 + thetaQ[i]
        RQtrue = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,1.0]])
        stress_lab[i] = np.reshape(RQtrue@np.reshape(stress[i],(3,3))@np.transpose(RQtrue),9)
        stress_til_lab[i] = np.reshape(RQtrue@np.reshape(stress_til[i],(3,3))@np.transpose(RQtrue),9)

        phaseQ = -pred_phase[i]/2.0 + thetaQ[i]
        RQpred = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,1.0]])
        stress_hattil_lab[i] = np.reshape(RQpred@np.reshape(stress_hattil[i],(3,3))@np.transpose(RQpred),9)

    # stress_lab = stress
    # stress_til_lab = stress_til
    # stress_hattil_lab = stress_hattil

    fig,axs = plt.subplots(nrows=4,ncols=1,figsize=(4,4),sharex=True,sharey=False,dpi=300)
    # fig = plt.figure(figsize=(10,3),dpi=200)
    ax = axs[0]
    ax.plot(tspan,stress_lab[:,0],'-',label='Data')
    # plt.plot(tspan,stress_til_lab[:,0],'--',label=r'No time int')
    ax.plot(tspan,stress_hattil_lab[:,0],':',label=r'SIMPLE')
    ax.legend()
    # plt.ylabel(r'$Q^T \sigma Q_{xx}$')
    ax.set_ylabel(r'$\sigma_{xx} / \eta_s \phi$')
    # plt.xlabel(r'$t$')
    # plt.xlabel(r'$\mathcal{T}$')
    # plt.title('a) '+r'$\sigma_{xx}$')

    # ax = plt.subplot(142)
    ax = axs[1]
    ax.plot(tspan,stress_lab[:,1],'-',label='Data')
    # plt.plot(tspan,stress_til_lab[:,1],'--',label=r'No time int')
    ax.plot(tspan,stress_hattil_lab[:,1],':',label=r'SIMPLE')
    # plt.ylabel(r'$Q^T \sigma Q_{xy}$')
    ax.set_ylabel(r'$\sigma_{xy} / \eta_s \phi$')
    # plt.xlabel(r'$\mathcal{T}$')
    # plt.title('b) '+r'$\sigma_{xy}$')

    # ax = plt.subplot(143)
    ax = axs[2]
    ax.plot(tspan,stress_lab[:,4],'-',label='Data')
    # plt.plot(tspan,stress_til_lab[:,4],'--',label=r'No time int')
    ax.plot(tspan,stress_hattil_lab[:,4],':',label=r'SIMPLE')
    # plt.ylabel(r'$Q^T \sigma Q_{yy}$')
    ax.set_ylabel(r'$\sigma_{yy} / \eta_s \phi$')
    # plt.title('c) '+r'$\sigma_{yy}$')
    # plt.xlabel(r'$\mathcal{T}$')

    # ax = plt.subplot(144)
    ax = axs[3]
    ax.plot(tspan,stress_lab[:,8],'-',label='Data')
    # plt.plot(tspan,stress_til_lab[:,8],'--',label=r'No time int')
    ax.plot(tspan,stress_hattil_lab[:,8],':',label=r'SIMPLE')
    # plt.ylabel(r'$Q^T \sigma Q_{zz}$')
    ax.set_ylabel(r'$\sigma_{zz} / \eta_s \phi$')
    ax.set_xlabel(r'$\mathcal{T}$')
    # plt.title('d) '+r'$\sigma_{zz}$')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(open('interp/stress_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))
    pickle.dump([tspan,stress_lab,stress_til_lab,stress_hattil_lab],open('interp/stress_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'wb'))

    nrows = 3
    ncols = 3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(3,2.75),sharex=True,sharey='row')

    count = 0
    for ax in axs.ravel():
        # print(ax)
        ax.plot(tspan,a[:,count],'o',markersize=3,label='Data')
        ax.plot(tspan,ahat[:,count],'s',markersize=1)
        ax.plot(tspan,atil[:,count],'s',fillstyle='none',markersize=3,markeredgewidth=0.5,label='AE')
        if count == 0 or count == 3 or count == 6:
            ax.set_ylabel(r'$a_i$')
        if count >= 6:
            ax.set_xlabel(r'$\mathcal{T}$')
            ax.set_xticks([0.0,0.3])
        if count == 0:
            ax.text(0.15, 0.1, r'$a_1$', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            ax.legend(frameon=False,handletextpad=0.1,loc=6, bbox_to_anchor=(-0.2, 0.2, 0.5, 0.5))
        else:
            ax.text(0.15, 0.85, r'$a_{%d}$'%(count+1), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        count = count + 1

    # fig.supxlabel(r'$t$')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig(open('interp/pca_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # exit()

    Irot = np.zeros(Ifill.shape)
    for i in range(n_win):   
        Irot[i] = ndimage.rotate(Ifill[i],phase_t[i]/2.0*180.0/np.pi - thetaQ[i]*180.0/np.pi,reshape=False)
        # Itilf[i] = ndimage.rotate(Itilf[i],phase_t[i]/2.0*180.0/np.pi - thetaQ[i]*180.0/np.pi,reshape=False)

    Ihtr = np.zeros(Ihattilf.shape)
    for i in range(n_win):   
        Ihtr[i] = ndimage.rotate(Ihattilf[i],pred_phase[i]/2.0*180.0/np.pi - thetaQ[i]*180.0/np.pi,reshape=False)

    Qmag = np.reshape(Qmag,(nqi,nqi))
    Irot = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, Irot)
    Itilf = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, Itilf)
    Ihtr = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, Ihtr)

    Iq_norm = np.mean(np.sqrt(IQa**2))

    Irnz = np.reshape(Irot,(Irot.shape[0],nqi*nqi))
    Irnz = Irnz[:,ind]
    Irnz = np.squeeze(Irnz)
    print(Irnz.shape)
    Ihnz = np.reshape(Ihtr,(Ihtr.shape[0],nqi*nqi))
    Ihnz = Ihnz[:,ind]
    Ihnz = np.squeeze(Ihnz)
    err_lab = np.mean(np.sqrt((Irnz - Ihnz)**2)/Iq_norm,axis=1)
    err_ali = np.mean(np.sqrt((IQa - Ihattil)**2)/Iq_norm,axis=1)
    err_lat = np.mean(np.sqrt((pred_y - h)**2),axis=1)
    err_ph = np.sqrt((phase_t - pred_phase)**2)
    # print(err_lab.shape)
    fig = plt.figure(figsize=(12,3),dpi=300)

    ax = plt.subplot(131)
    plt.plot(tspan,err_lab,'o-')
    plt.ylabel(r'$|I - \tilde{\hat{I}}|/|I|$')
    plt.xlabel(r'$\tilde{t}$')

    ax = plt.subplot(132)
    plt.plot(tspan,err_ali,'o-')
    plt.ylabel(r'$|I_{\alpha} - \tilde{\hat{I}}_{\alpha}|/|I|$')
    plt.xlabel(r'$\tilde{t}$')

    ax = plt.subplot(133)
    plt.plot(tspan,err_lat,'o-',label=r'$h$')
    plt.plot(tspan,err_ph,'s--',label=r'$\alpha$')
    plt.legend()
    plt.ylabel(r'$|h - \hat{h}|, |\alpha - \hat{\alpha}|$')
    plt.xlabel(r'$\tilde{t}$')

    plt.tight_layout()
    plt.savefig(open('interp/err_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))

    pickle.dump([tspan,err_lab,err_ali,err_lat,err_ph],open('interp/err_%.2f_%.2f_%d.p'%(FRR,Q2,strm),'wb'))

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
    plt.savefig(open('interp/Qphase_enc_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # plt.show()

    # vmin = np.min(Itil)
    # vmax = np.max(Itil)
    # errmin = np.min(np.sqrt((I - Ihattil)**2))
    # errmax = np.max(np.sqrt((I - Ihattil)**2))
    errmin = 1e-4
    errmax = 10.0
    # print(IQa[-1,-1])
    # vmin = np.min(Ihattil)
    # if vmin < 0:
    #     vmin = np.min(I)
    # print(vmin)
    vmin = 0.01

    mymap=matplotlib.cm.get_cmap('jet')

    # fig,axs = plt.subplots(nrows=3,ncols=1,figsize=(2.25,4.61),sharex=True,sharey=False,dpi=300)
    # ax0 = axs[0]
    # # plt.plot(t,flow_type)
    # ax0.plot(tspan,flow_type)
    # # ax0.set_xlabel('Time')
    # # ax0.set_ylabel('Flow type parameter '+r'$\Lambda$')
    # ax0.set_ylabel(r'$\Lambda$')

    # ax00 = axs[1]
    # # plt.plot(t,G)
    # ax00.plot(tspan,G)
    # # plt.xlabel('Time')
    # # ax00.set_ylabel('Flow strength G')
    # ax00.set_ylabel(r'$G/D_r$')

    # ax000 = axs[2]
    # # plt.plot(t,G)
    # ax000.plot(tspan,err_lab,'o-',markersize=4)
    # ax000.set_xlabel(r'$\mathcal{T}$')
    # ax000.set_ylabel(r'$|I - \tilde{\hat{I}}|/|I|$')
    # ax000.set_xlim(0,)

    # plt.tight_layout()
    # plt.savefig(open('interp/snaps1_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))

    # fig,axs = plt.subplots(nrows=3,ncols=2,figsize=(4.5,5.0625),sharex=True,sharey=True,dpi=300)
    # # fig,axs = plt.subplots(nrows=3,ncols=2,figsize=(4.75,5.5),sharex=True,sharey=True,dpi=300)

    # window = 1
    # # ax1 = plt.subplot(321)
    # ax1 = axs[0,0]
    # # cax1 = ax1.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax1 = ax1.pcolormesh(qxlist,qylist,Ifill[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # # cax1 = ax1.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax1.set_aspect('equal')
    # # ax.set_xlabel(r'$q_x$')
    # ax1.set_ylabel(r'$q_y$')
    # # ax1.set_title('Data '+r'$\mathcal{T} = %.2f$'%(dt*window))
    # ax1.set_title('True Data')
    # ax1.set_yticks([-0.1,0.0,0.1])

    # # ax2 = plt.subplot(322)
    # ax2 = axs[0,1]
    # cax2 = ax2.pcolormesh(qxlist,qylist,Ihtr[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax2.set_aspect('equal')
    # # ax.set_xlabel(r'$q_x$')
    # # ax.set_ylabel(r'$q_y$')
    # # ax2.set_title('NODE '+r'$\mathcal{T} = %.2f$'%(dt*window))
    # ax2.set_title('Prediction')
    # # ax2.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # # ax2.set_yticklabels(['','','','',''])
    # # cb2 = fig.colorbar(cax2,ax=ax2)
    # # cb2.ax.set_title(r'$I_R(q)$')

    # window = 24
    # # ax3 = plt.subplot(323)
    # ax3 = axs[1,0]
    # # cax3 = ax3.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax3 = ax3.pcolormesh(qxlist,qylist,Ifill[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # # cax3 = ax3.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax3.set_aspect('equal')
    # ax3.set_ylabel(r'$q_y$')
    # # ax3.set_title('Data '+r'$\mathcal{T} = %.2f$'%(dt*window)))
    # # ax3.set_yticks([-0.1,-0.05,0.0,0.05,0.1])

    # # ax4 = plt.subplot(324)
    # ax4 = axs[1,1]
    # cax4 = ax4.pcolormesh(qxlist,qylist,Ihtr[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax4.set_aspect('equal')
    # # ax4.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # # ax4.set_yticklabels(['','','','',''])
    # # ax4.set_title('NODE '+r'$\mathcal{T} = %.2f$'%(dt*window)))

    # window = n_win - 1
    # # ax5 = plt.subplot(325)
    # ax5 = axs[2,0]
    # # cax5 = ax5.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax5 = ax5.pcolormesh(qxlist,qylist,Ifill[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # # cax5 = ax5.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax5.set_aspect('equal')
    # ax5.set_xlabel(r'$q_x$')
    # ax5.set_ylabel(r'$q_y$')
    # # ax5.set_title('Model lab frame')
    # # ax5.set_title('Data '+r'$\mathcal{T} = %.2f$'%(dt*window)))
    # # ax5.set_yticks([-0.1,-0.05,0.0,0.05,0.1])

    # # ax6 = plt.subplot(326)
    # ax6 = axs[2,1]
    # cax6 = ax6.pcolormesh(qxlist,qylist,Ihtr[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # ax6.set_aspect('equal')
    # ax6.set_xlabel(r'$q_x$')
    # # ax6.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # # ax6.set_yticklabels(['','','','',''])
    # # ax6.set_title('NODE '+r'$\mathcal{T} = %.2f$'%(dt*window))

    # cb = fig.colorbar(cax6, ax=axs.ravel().tolist(),aspect=40)
    # cb.ax.set_title(r'$I(q)$')

    # # plt.tight_layout()
    # plt.savefig(open('interp/snaps_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))

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
        cb2 = fig.colorbar(cax2,ax=ax2)
        cb2.ax.set_title(r'$I_R(q)$')

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
        cb5 = fig.colorbar(cax5,ax=ax5)
        cb5.ax.set_title(r'$I_R(q)$')

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

        if len(Ifill) < 100:
            anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(Ifill))
        else:
            nframes = 100
            frames = np.linspace(0,len(Ifill),num=nframes,endpoint=False,dtype=int)
            print(frames)
            anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=frames)
        anim.save('interp/Qphase_FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))
        # plt.show()


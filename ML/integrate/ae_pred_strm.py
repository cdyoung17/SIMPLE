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
parser.add_argument('--data_size', type=int, default=95181)  #IC from the simulation
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

# Use the trained model with SciPy integrators to reduce cost of storing the gradients
class SciPyODE(nn.Module):
    def __init__(self,dh,alpha):
        super(SciPyODE, self).__init__()
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
        # self.net = nn.Sequential(
        #     nn.Linear(dh+6, 256),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.ReLU(),
        #     nn.Linear(256, dh),
        # )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        g = gfunc(t)
        hg = np.concatenate((y,g))
        hg = torch.tensor(hg)
        hg = hg.type(torch.FloatTensor)
        return self.net(hg).detach().numpy() - self.alpha*y

        # E = Efunc(t)
        # hE = np.concatenate((y,E))
        # hE = torch.tensor(hE)
        # hE = hE.type(torch.FloatTensor)
        # return self.net(hE).detach().numpy() - self.alpha*y

class autoencoder(nn.Module):
    def __init__(self, ambient_dim=30, code_dim=20, filepath='testae'):
        super(autoencoder, self).__init__()
        
        self.ambient_dim = ambient_dim
        self.code_dim = code_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.ambient_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.code_dim),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False),
            nn.Linear(self.code_dim, self.code_dim, bias=False))
        
        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.ambient_dim))
        
    def forward(self,x):
        code = self.encoder(x)
        xhat = self.decoder(code)
        return xhat
    
    def encode(self, x):
        code = self.encoder(x)
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
    ae_model = 20

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
    PCA_path = '../../BD_FFoRM/c_rods/training_output/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/pca_%d.p'%(nbins,nqi,qlim,L,R,data_size)
    U,S = pickle.load(open(PCA_path,'rb'))
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
    # E = Gd
    E = np.reshape(E,(E.shape[0],9))
    Efull = E
    Gd = np.reshape(Gd,(Gd.shape[0],9))

    red_sym_ind = np.array([0,1,2,4,5,8])
    E = E[:,red_sym_ind]
    # Gd = Gd[:,red_sym_ind]
    stress = stress[:,red_sym_ind]

    # Create interpolation function for continuous evaluation of E for NODE input
    gfunc = interp1d(tspan, Gd, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
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
        Gd = Gd[:-1]
        tspan = tspan[:-1]

    # cutoff = int(0.1*len(I))
    # I = I[cutoff:-cutoff]
    # stress = stress[cutoff:-cutoff]
    # E = E[cutoff:-cutoff]
    # Gd = Gd[cutoff:-cutoff]
    # tspan = tspan[cutoff:-cutoff]

    # Project onto truncated PCA modes
    a = I @ U[:,:dPCA]
    print(a.shape)
    print(dPCA)
    kernel_size = np.max((1,int(len(a)/20)))
    a = rolling_mean_along_axis(a,kernel_size,0)

    # a_mean,a_std = pickle.load(open('a_stats_v2_%d.p'%dPCA,'rb'))
    # anp = (anp - a_mean[np.newaxis,:])/a_std
    # [M,N] = anp.shape

    # DNN AE
    model = autoencoder(ambient_dim=dPCA).to(device)
    ae_path = '../autoencoder/dPCA%d/model%d/'%(dPCA,ae_model)
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
    print(h.shape)

    wdir = 'dh%d/dPCA%d/arch%d_tau%d_alpha%.1e/model%d'%(dh,dPCA,arch,batch_time,alpha,modl)

    os.chdir(wdir)

    func_sp = SciPyODE(dh,alpha)
    # func_sp.load_state_dict(torch.load('state_dict.pt'))
    func_sp.load_state_dict(torch.load('chp/sd.pt'))
    func_sp.eval()

    nt = len(tspan)
    sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],h[0],t_eval=tspan,max_step=dt)
    pred_y = np.transpose(sol['y'])

    bhat = pred_y @ np.transpose(u[:,:dh])
    bhat = T.tensor(bhat, dtype=T.float).to(device)
    ahat = model.decode(bhat)
    ahat = ahat.detach().numpy()
    ahat = ahat*a_std + a_mean[np.newaxis,:]
    a = a*a_std + a_mean[np.newaxis,:]
    Ihattil = ahat @ np.transpose(U[:,:dPCA])
    I = np.reshape(I,(I.shape[0],nqi,nqi))
    Ihattil = np.reshape(Ihattil,(Ihattil.shape[0],nqi,nqi))

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

    plt.tight_layout()
    plt.savefig(open('enc_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # plt.show()

    # vmin = np.min(Itil)
    # vmax = np.max(Itil)
    # errmin = np.min(np.sqrt((I - Ihattil)**2))
    # errmax = np.max(np.sqrt((I - Ihattil)**2))
    errmin = 1e-4
    errmax = 5e2
    # vmin = np.min(Ihattil)
    # if vmin < 0:
        # vmin = np.min(I)
    vmin = 0.01
    # print(vmin)

    mymap=matplotlib.cm.get_cmap('jet')
    nrows = 2
    ncols = 3
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(8,4.1),sharex=True,sharey=True,dpi=300)

    count = 0
    windows = [2,32,57,2,32,57]
    for ax in axs.ravel():
        # amin = np.min(mode)
        # print(count,amin)
        # cax = ax.pcolormesh(qxlist,qylist,mode,norm=colors.LogNorm(),cmap=mymap,shading='gouraud')
        if count < 3:
            cax = ax.pcolormesh(qxlist,qylist,I[windows[count]],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
            ax.set_title(r'$\tilde{t}_%d = %.2f$'%(count+1,tspan[windows[count]]))
        else:
            cax = ax.pcolormesh(qxlist,qylist,Ihattil[windows[count]],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        # cax = ax.pcolormesh(qxlist,qylist,mode,cmap=mymap,shading='gouraud')
        # ax.set_title('Mode %d'%(start+count+1))
        # ax.set_title(r'$U_%d$'%(start+count))
        ax.set_aspect('equal')
        if count >= 3:
            ax.set_xlabel(r'$q_x \ [\AA^{-1}]$')
        if count == 0 or count == 3:
            ax.set_ylabel(r'$q_y \ [\AA^{-1}]$')
        # cb = fig.colorbar(cax,ax=ax)
        # cb.ax.set_title(r'$a(q)$')

        count = count + 1

    fig.subplots_adjust(wspace=0.3)
    ax.set_yticks([-0.1,0,0.1])
    ax.set_xticks([-0.1,0,0.1])
    cb = fig.colorbar(cax, ax=axs.ravel().tolist())
    cb.ax.set_title(r'$I(q)$')
    plt.savefig(open('snaps_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))


    if args.movie == 1:
        tmov = np.linspace(0,30,len(I))
        # mymap=matplotlib.cm.get_cmap('inferno')

        fig = plt.figure(figsize=(10.5,3),dpi=300)
        ax3 = plt.subplot(131)
        cax3 = ax3.pcolormesh(qxlist,qylist,I[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        ax3.set_aspect('equal')
        ax3.set_xlabel(r'$q_x$')
        ax3.set_ylabel(r'$q_y$')
        cb3 = fig.colorbar(cax3,ax=ax3)
        cb3.ax.set_title(r'$I(q)$')

        ax4 = plt.subplot(132)
        cax4 = ax4.pcolormesh(qxlist,qylist,Ihattil[0],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
        # cax4 = ax4.pcolormesh(qxlist,qylist,Ihattil[0],norm=colors.SymLogNorm(linthresh=1e-5, linscale=0.03, base=10),cmap=mymap,shading='gouraud')
        ax4.set_aspect('equal')
        ax4.set_xlabel(r'$q_x$')
        # ax4.set_ylabel(r'$q_y$')
        # cb4 = fig.colorbar(cax4,ax=ax4)
        # cb4.ax.set_title(r'$\hat{\tilde{I}}(q)$')

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

    dPCAs = 30
    ahat = ahat[:,:dPCAs]
    aE = np.concatenate((ahat,E),axis=1)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(aE.shape[1])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128,activation='relu'))
    model.add(keras.layers.Dense(units=128,activation='relu'))
    model.add(keras.layers.Dense(stress.shape[1]))

    model.compile(loss='mse', optimizer='adam')

    # # Save the architecture summary for quick viewing
    # with open('model_summary.txt', 'w') as f:
    #     model.summary(print_fn=lambda x: f.write(x + '\n'))

    archs = 2
    regs = 0.0001
    models = 0
    stress_path = '../../../../../stress/dPCA%d/arch%d_reg%.6f/model%d/'%(dPCAs,archs,regs,models)
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

    fig = plt.figure(figsize=(16,3),dpi=200)

    ax = plt.subplot(141)
    ax.plot(tspan,Gd[:,0],'o-',markersize=1,label='gradient')
    ax.plot(tspan,Efull[:,0],'s--',markersize=1,label='rate of strain')
    # ax.plot(t,E[:,0],label='E_{00}')
    plt.legend()
    ax.set_ylabel(r'$E_{xx},\nabla v_{xx}$')

    ax = plt.subplot(142)
    ax.plot(tspan,Gd[:,1],'o-',markersize=1)
    ax.plot(tspan,Efull[:,1],'s--',markersize=1)
    ax.set_ylabel(r'$E_{xy},\nabla v_{xy}$')

    ax = plt.subplot(143)
    ax.plot(tspan,Gd[:,3],'o-',markersize=1)
    ax.plot(tspan,Efull[:,3],'s--',markersize=1)
    ax.set_ylabel(r'$E_{yx},\nabla v_{yx}$')

    ax = plt.subplot(144)
    ax.plot(tspan,Gd[:,4],'o-',markersize=1)
    ax.plot(tspan,Efull[:,4],'s--',markersize=1)
    ax.set_ylabel(r'$E_{yy},\nabla v_{yy}$')
    ax.set_xlabel(r'$t$')

    plt.tight_layout()

    plt.savefig(open('flow_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))


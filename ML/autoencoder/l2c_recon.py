#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import argparse

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.tri as mtri
from matplotlib import cm

import scipy
from scipy.ndimage import uniform_filter1d
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.io import savemat
from scipy import special

import torch
import torch.nn as nn
import torch.optim as optim
import torch as T
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser('movie')
parser.add_argument('--FRR', type=float, default=-0.7) 
parser.add_argument('--Q2', type=float, default=10.0) 
parser.add_argument('--strm', type=int, default=4)
parser.add_argument('--ntraj', type=int, default=100000)
parser.add_argument('--avg_win', type=float, default=0.01)
parser.add_argument('--nqi', type=int, default=100)
parser.add_argument('--qlim', type=float, default=0.1)
parser.add_argument('--nbins', type=int, default=200)
parser.add_argument('--dPCA', type=int, default=50)
parser.add_argument('--dh', type=int, default=3)
parser.add_argument('--M', type=int, default=95119)
parser.add_argument('--nlin', type=int, default=4) # Number of linear layers in latent space
parser.add_argument('--cd', type=int, default=20) # Linear layer width
parser.add_argument('--model', type=int, default=0)
parser.add_argument('--movie', type=int, default=0)
parser.add_argument('--dir', type=str, default='') # Number of snapshots
# parser.add_argument('--arch', type=int, default=0) # Architecture label
# parser.add_argument('--model', type=int, default=0) # Model #
# parser.add_argument('--reg', type=float, default=0.001) # Model #
# parser.add_argument('--M', type=int, default=95181) # Number of snapshots
# parser.add_argument('--load', type=int, default=0) # Number of snapshots
args = parser.parse_args()

class autoencoder(nn.Module):
    def __init__(self, ambient_dim=30, code_dim=20, nlin = 4, filepath='testae'):
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
            # nn.Linear(self.code_dim, self.code_dim, bias=False),
            # nn.Linear(self.code_dim, self.code_dim, bias=False),
            # nn.Linear(self.code_dim, self.code_dim, bias=False),
            # nn.Linear(self.code_dim, self.code_dim, bias=False))

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
            code = l[1](code)
        return code
    
    def decode(self, code):
        xhat = self.decoder(code)
        return xhat

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    

device = T.device("cuda" if T.cuda.is_available() else "cpu")

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

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    FRR = args.FRR
    Q2 = args.Q2
    strm = args.strm
    avg_win = args.avg_win
    ntraj = args.ntraj
    nqi = args.nqi
    qlim = args.qlim
    nbins = args.nbins
    dPCA = args.dPCA
    dh = args.dh
    modl = args.model
    movie = args.movie
    nlin = args.nlin
    code_dim = args.cd
    samp_rate = 1

    data = np.loadtxt('../../BD_FFoRM/Gamma_input/FRR_%.2f/Q2_%.2f/strm_%d.txt'%(FRR,Q2,strm),skiprows=2)
    tflow = data[:,0]
    flow_type = data[:,1]
    flow_strength = data[:,2]
    Gdot = data[:,3:]
    Gdot = np.reshape(Gdot,(Gdot.shape[0],3,3))
    Dr = 2.0
    tflow = tflow*Dr
    Gdot = Gdot/Dr
    dt = 0.0005
    tend = tflow[-1]
    tmax = int(tend/dt)
    # print('No. time steps ',tmax)
    ts = np.arange(0,tmax)*dt
    FTP_o = np.zeros(tmax)
    G_o = np.zeros(tmax)
    f_step = 0
    t_elap = 0.0

    steps_p_win = int(avg_win/(dt*samp_rate))
    print('Time steps per window %d'%steps_p_win)
    n_win = math.ceil(tend/avg_win)
    print('Number of windows %d'%n_win)

    for tj in range(tmax):

        if t_elap > tflow[f_step]:
            f_step = f_step + 1

        FTP_o[tj] = flow_type[f_step]
        G_o[tj] = flow_strength[f_step]
        t_elap += dt

    FTP_o = FTP_o[::steps_p_win]
    G_o = G_o[::steps_p_win]
    ts = ts[::steps_p_win]

    L = 1680.0;
    R = 34.0;
    volfrac=0.1 # volume fraction of cylinders
    dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
    bkgd = 0.0001 # background scattering in cm^-1

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
    print(ind)
    gamma = 0.0

    # Cylinders
    # data_path = '../../BD_FFoRM/c_rods/training_output/%s_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(args.dir,nbins,nqi,qlim,L,R)
    data_path = '../../BD_FFoRM/c_rods/training_output/l2c_smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    # I,ornt,cum_weight = pickle.load(open(data_path+'QRI_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    I,Q,phase_t,Gd,E = pickle.load(open(data_path+'alignedI_labphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    # I2,Q,phase_t,Gd,E = pickle.load(open(data_path+'IQphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    # I = 1.0e8*np.pi*R**2*L*volfrac*dsld**2*P/cum_weight[:,np.newaxis,np.newaxis] + bkgd
    # I = np.reshape(I,(I.shape[0],I.shape[1]*I.shape[2]))
    # kernel_size = np.max((1,int(len(I)/20)))
    # I = rolling_mean_along_axis(I,kernel_size,0)

    if np.isnan(I[-1,-1]) or np.isinf(I[-1,-1]):
        n_win -= 1

    # Load PCA modes
    n_load = args.M
    # PCA_path = '../../BD_FFoRM/c_rods/training_output/smooth_na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    # U,S = pickle.load(open(PCA_path+'QRpca_%d.p'%n_load,'rb'))
    U,S = pickle.load(open(data_path+'Qphase_pca_%d.p'%n_load,'rb'))
    print(I.shape)
    print(U.shape)
    print(S[0:30])
    a = I @ U[:,:dPCA]
    print(a.shape)

    # Load autoencoder
    wdir = 'dPCA%d/%s/rmodel%d/'%(dPCA,args.dir,modl)
    print(wdir)
    os.chdir(wdir)

    a_mean,a_std = pickle.load(open('a_stats_v2_%d.p'%dPCA,'rb'))
    aT = (a - a_mean[np.newaxis,:])/a_std
    # print(ts.shape)

    # DNN build model
    model = autoencoder(ambient_dim=dPCA,code_dim=code_dim,nlin=nlin).to(device)
    if(os.path.exists('IRMAE_AE.pt')) == 1:
        model.load_state_dict(T.load('IRMAE_AE.pt'))
    else:
        model.load_state_dict(T.load('IRMAE_AE_chp.pt'))

    # DNN encode and decode
    aT = T.tensor(aT, dtype=T.float).to(device)
    b = model.encode(aT)
    b = b.detach().numpy()
    print(b.shape)
    u,s,v = pickle.load(open('code_svd.p','rb'))
    print(s)
    h = b @ u[:,:dh]
    print(h.shape)
    bhat = h @ np.transpose(u[:,:dh])
    print(bhat.shape)
    err_pca = np.mean((b-bhat)**2)
    print('SVD-PCA error',err_pca)
    bhat = T.tensor(bhat, dtype=T.float).to(device)
    ahat = model.decode(bhat)
    ahat = ahat.detach().numpy()
    err_ae = np.mean((ahat - (a - a_mean[np.newaxis,:])/a_std)**2)
    print(ahat.shape)
    print('AE norm error',err_ae)

    nrows = 5
    ncols = 5
    # fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(10,10),sharex=True,sharey='row',dpi=300)
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,10),sharex=True,dpi=300)

    # count = 0
    # for ax in axs.ravel():
    #     # print(ax)
    #     if count < 4:
    #         ax.plot(ts,aT[:,count],'o',markersize=2,label='Data')
    #         ax.plot(ts,ahat[:,count],'s',fillstyle='none',markersize=2,label='AE',markeredgewidth=0.5)
    #     else:
    #         ax.plot(ts,h[:,count-4],'^',markersize=2,label='Data',color='black')
    #     # ax.set_ylabel(r'$a_{%d}$'%(count+1))
    #     if count == 0:
    #         ax.set_ylabel(r'$a_i$')
    #         ax.legend()
    #     if count == 4:
    #         ax.set_ylabel(r'$h_i$')
    #     # ax.set_title('Mode %d'%(count+1))
    #     if count >= 4:
    #         ax.set_xlabel(r'$t$')
    #     count = count + 1
    count = 0
    for ax in axs.ravel():
        # print(ax)
        ax.plot(ts,aT[:,count],'o',markersize=2,label='Data')
        ax.plot(ts,ahat[:,count],'s',fillstyle='none',markersize=2,label='AE',markeredgewidth=0.5)
        # ax.set_ylabel(r'$a_{%d}$'%(count+1))
        count = count + 1

    fig.supxlabel(r'$t$')
    plt.tight_layout()
    plt.savefig(open('pca_%.2f_%.2f_%d.png'%(FRR,Q2,strm),'wb'))
    # plt.show()
    # exit()

    # PCA project back to full space
    ahat = ahat*a_std + a_mean[np.newaxis,:]
    err_ae = np.mean((ahat - a)**2)
    print('AE error',err_ae)
    Itil = a @ np.transpose(U[:,:dPCA])
    Itilhat = ahat @ np.transpose(U[:,:dPCA])
    err_tot_pca = np.mean((Itil - I)**2)
    err_tot_ae = np.mean((Itilhat - I)**2)
    print('AE total error',err_tot_ae)
    print('PCA error',err_tot_pca)    
    err_pca_t = np.mean((Itil - I)**2,axis=1)
    err_ae_t = np.mean((Itilhat - I)**2,axis=1)

    Ifill = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Ifill[i,ind] = I[i]
    I = np.reshape(Ifill,(Ifill.shape[0],nqi,nqi))

    Ifill = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Ifill[i,ind] = Itil[i]
    Itil = np.reshape(Ifill,(Ifill.shape[0],nqi,nqi))

    Ifill = np.zeros((n_win,nqi*nqi))
    for i in range(n_win):
        Ifill[i,ind] = Itilhat[i]
    Itilhat = np.reshape(Ifill,(Ifill.shape[0],nqi,nqi))

    vmax = 600
    vmin = 1e-3
    window = 10

    # fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(3.1,5.9),dpi=300,sharex=True)

    # ax = axs[0]
    # cax = ax.pcolormesh(qxlist,qylist,I[window],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=matplotlib.cm.get_cmap('jet'),shading='gouraud')
    # ax.set_aspect('equal')
    # # ax.set_xlabel(r'$q_x \ [\AA^{-1}]$')
    # ax.set_ylabel(r'$q_y \ [\AA^{-1}]$')
    # ax.set_yticks([-0.1,0,0.1])
    # ax.set_title('Data')
    # # cb = fig.colorbar(cax,ax=ax)
    # # cb.ax.set_title(r'$I(q)$')

    # ax = axs[1]
    # cax = ax.pcolormesh(qxlist,qylist,Itilhat[window],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=matplotlib.cm.get_cmap('jet'),shading='gouraud')
    # ax.set_aspect('equal')
    # ax.set_title('Reconstruction')
    # ax.set_xlabel(r'$q_x \ [\AA^{-1}]$')
    # ax.set_ylabel(r'$q_y \ [\AA^{-1}]$')
    # ax.set_yticks([-0.1,0,0.1])
    # ax.set_xticks([-0.1,0,0.1])

    # # fig.subplots_adjust(bottom=0.8)
    # # cbar_ax = fig.add_axes([, 0.15, 0.05, 0.7])
    # # fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')

    # cb = fig.colorbar(cax,ax=axs.ravel().tolist(),orientation='horizontal',fraction=0.02,pad=0.12)
    # cb.ax.text(900.0,1e-3,r'$I(q)$')

    # # plt.tight_layout()
    # plt.savefig(open('recon_snap_%.2f_%.2f_%d_%d.png'%(FRR,Q2,strm,window),'wb'))

    fig = plt.figure(figsize=(12,6),dpi=300)
    ax = plt.subplot(241)
    plt.plot(ts,h[:,0],'o-')
    plt.ylabel(r'$h_0$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(242)
    plt.plot(ts,h[:,1],'o-')
    plt.ylabel(r'$h_1$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(243)
    plt.plot(ts,h[:,2],'o-')
    plt.ylabel(r'$h_2$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(244)
    plt.plot(ts,h[:,3],'o-')
    plt.ylabel(r'$h_3$')
    plt.xlabel(r'$t$')

    if dh > 4:
        ax = plt.subplot(245)
        plt.plot(ts,h[:,4],'o-')
        plt.ylabel(r'$h_4$')
        plt.xlabel(r'$t$')

    if dh > 5:
        ax = plt.subplot(246)
        plt.plot(ts,h[:,5],'o-')
        plt.ylabel(r'$h_5$')
        plt.xlabel(r'$t$')

    if dh > 6:
        ax = plt.subplot(247)
        plt.plot(ts,h[:,6],'o-')
        plt.ylabel(r'$h_6$')
        plt.xlabel(r'$t$')

    if dh > 7:
        ax = plt.subplot(248)
        plt.plot(ts,h[:,7],'o-')
        plt.ylabel(r'$h_7$')
        plt.xlabel(r'$t$')

    plt.tight_layout()
    # plt.savefig(open('../../strm_recon/dPCA%d_tr%d_FRR_%.2f_Q2_%.2f_strm_%d.png'%(dPCA,modl,FRR,Q2,strm),'wb'))
    plt.savefig(open('FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))

    fig = plt.figure(figsize=(4,3),dpi=300)
    ax = plt.subplot(111)
    plt.semilogy(ts,err_pca_t,'o-',label='PCA')
    plt.semilogy(ts,err_ae_t,'s--',label='PCA+AE')
    plt.legend()
    plt.ylabel(r'$|I - I_{pred}|^2$')
    plt.xlabel(r'$t$')

    plt.tight_layout()
    # plt.savefig(open('../../strm_recon/err_dPCA%d_tr%d_FRR_%.2f_Q2_%.2f_strm_%d.png'%(dPCA,modl,FRR,Q2,strm),'wb'))
    plt.savefig(open('err_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))

    if movie==1:
        t = np.linspace(0,10,n_win)
        # mymap=matplotlib.cm.get_cmap('inferno')
        mymap=matplotlib.cm.get_cmap('jet')

        fig = plt.figure(figsize=(12,6),dpi=300)

        # ax1 = plt.subplot(141)
        # ax1.plot(tflow,flow_type)
        # pt_ftp, = ax1.plot(ts[0],FTP_o[0],'o')
        # # print(pt_ftp[0])
        # # exit()
        # ax1.set_xlabel('Time (s)')
        # ax1.set_ylabel('Flow type parameter '+r'$\Lambda$')

        # ax2 = plt.subplot(142)
        # ax2.plot(tflow,flow_strength)
        # pt_g, = ax2.plot(ts[0],G_o[0],'o')
        # ax2.set_xlabel('Time (s)')
        # ax2.set_ylabel('Flow strength G (1/s)')

        # # anim = matplotlib.animation.FuncAnimation(fig, animate_FTP, interval=100, frames=len(ts))
        # # anim.save(wdir+'strm_movie.mp4')

        vmax = 600
        vmin = 1e-3

        ax3 = plt.subplot(231)
        cax3 = ax3.pcolormesh(qxlist,qylist,I[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax3.set_aspect('equal')
        ax3.set_xlabel(r'$q_x$')
        ax3.set_ylabel(r'$q_y$')
        cb3 = fig.colorbar(cax3,ax=ax3)
        cb3.ax.set_title(r'$I(q)$')
        plt.tight_layout()

        ax5 = plt.subplot(232)
        cax5 = ax5.pcolormesh(qxlist,qylist,Itil[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax5.set_aspect('equal')
        ax5.set_xlabel(r'$q_x$')
        ax5.set_ylabel(r'$q_y$')
        cb5 = fig.colorbar(cax5,ax=ax5)
        cb5.ax.set_title(r'$\tilde{I}(q)$')
        plt.tight_layout()

        ax4 = plt.subplot(233)
        cax4 = ax4.pcolormesh(qxlist,qylist,Itilhat[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax4.set_aspect('equal')
        ax4.set_xlabel(r'$q_x$')
        ax4.set_ylabel(r'$q_y$')
        cb4 = fig.colorbar(cax4,ax=ax4)
        cb4.ax.set_title(r'$\tilde{\hat{I}}(q)$')
        plt.tight_layout()

        ax6 = plt.subplot(234)
        cax6 = ax6.pcolormesh(qxlist,qylist,np.sqrt((I[0] - Itil[0])**2),norm=colors.LogNorm(vmin=1e-3),cmap=mymap,shading='gouraud')
        ax6.set_aspect('equal')
        ax6.set_xlabel(r'$q_x$')
        ax6.set_ylabel(r'$q_y$')
        cb6 = fig.colorbar(cax6,ax=ax6)
        cb6.ax.set_title(r'$|I - \tilde{I}(q)|$')
        plt.tight_layout()

        ax7 = plt.subplot(235)
        cax7 = ax7.pcolormesh(qxlist,qylist,np.sqrt((I[0] - Itilhat[0])**2),norm=colors.LogNorm(vmin=1e-3),cmap=mymap,shading='gouraud')
        ax7.set_aspect('equal')
        ax7.set_xlabel(r'$q_x$')
        ax7.set_ylabel(r'$q_y$')
        cb7 = fig.colorbar(cax7,ax=ax7)
        cb7.ax.set_title(r'$|I - \tilde{\hat{I}}(q)|$')
        plt.tight_layout()

        # fig.colorbar(cax)

        def animate(i):
           # pt_ftp.set_ydata(FTP_o[i])
           # pt_ftp.set_xdata(ts[i])
           # pt_g.set_ydata(G_o[i])
           # pt_g.set_xdata(ts[i])
           # ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           cax3.set_array(I[i])
           cax4.set_array(Itilhat[i])
           cax5.set_array(Itil[i])
           cax6.set_array(np.sqrt((I[i] - Itil[i])**2))
           cax7.set_array(np.sqrt((I[i] - Itilhat[i])**2))
           ax3.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           ax4.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           ax5.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))

        anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(t))
        anim.save('FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))
        # plt.show()


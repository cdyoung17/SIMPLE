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
import parse
import scipy
from scipy import ndimage,misc

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
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--arch', type=int, default=0)
parser.add_argument('--traj', type=int, default=11)
# parser.add_argument('--flowType', type=float, default=0.0) 
# parser.add_argument('--Pe', type=float, default=10.0) 
# parser.add_argument('--movie', type=int, default=0)
parser.add_argument('--dir', type=str, default='l2c_smooth_sin') 
# parser.add_argument('--ae_model', type=int, default=0)
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1  

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
    # flowType = args.flowType
    # Pe = args.Pe
    # ae_model = args.ae_model

    wdir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d'%(dh,dPCA,args.dir,arch,batch_time,alpha,modl)
    os.chdir(wdir)

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
    Qmag = np.reshape(Qmag,(Qxnum,Qynum))

    flowType = 0.0
    flowType2 = 1.0

    Pelist = np.logspace(0,np.log10(50.0),10)
    low = np.array([0.1,0.5])
    Pelist = np.append(low,Pelist)
    print(Pelist)
    avg_win = np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.01,0.01,0.01])

    Pelist = Pelist[[0,5,7,9,10]]
    print(Pelist)
    Is_true = np.zeros((len(Pelist),Qxnum,Qynum))
    Is_pred = np.zeros((len(Pelist),Qxnum,Qynum))
    Ie_true = np.zeros((len(Pelist),Qxnum,Qynum))
    Ie_pred = np.zeros((len(Pelist),Qxnum,Qynum))
    count = 0
    for Pe in Pelist:
        Is_true[count], Is_pred[count] = pickle.load(open('extrap/ss_snap_%.2f_%.2e.p'%(flowType,Pe),'rb'))
        Is_true[count] = ndimage.rotate(Is_true[count],-45.0,reshape=False)
        Is_true[count] = np.where(Qmag[np.newaxis,:,:] < qlim, Is_true[count], 0.0)
        Is_pred[count] = ndimage.rotate(Is_pred[count],-45.0,reshape=False)
        Is_pred[count] = np.where(Qmag[np.newaxis,:,:] < qlim, Is_pred[count], 0.0)
        count += 1

    count = 0
    for Pe in Pelist:
        Ie_true[count], Ie_pred[count] = pickle.load(open('extrap/ss_snap_%.2f_%.2e.p'%(flowType2,Pe),'rb'))
        count += 1

    vmin = 0.01
    vmax = np.max(Ie_true[-1])
    mymap=matplotlib.cm.get_cmap('jet')
    fig,axs = plt.subplots(nrows=4,ncols=len(Pelist),figsize=(12,7),sharex=True,sharey=True,dpi=300)

    for i in range(len(Pelist)):
        ax = axs[0,i]
        cax = ax.pcolormesh(qxlist,qylist,Is_true[i],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        if i == 0:
            ax.set_ylabel(r'$q_y$')
            ax.text(-0.7, 0.5, 'Shear flow data', transform=ax.transAxes,rotation='vertical',horizontalalignment='center',verticalalignment='center')
        ax.set_title('Pe = %.1f'%Pelist[i])

    for i in range(len(Pelist)):
        ax = axs[1,i]
        cax = ax.pcolormesh(qxlist,qylist,Is_pred[i],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        if i == 0:
            ax.set_ylabel(r'$q_y$')
            ax.text(-0.7, 0.5, 'Shear flow SIMPLE', transform=ax.transAxes,rotation='vertical',horizontalalignment='center',verticalalignment='center')
        # ax.set_title('Pe = %.2f'%Pelist[i])

    for i in range(len(Pelist)):
        ax = axs[2,i]
        cax = ax.pcolormesh(qxlist,qylist,Ie_true[i],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax.set_aspect('equal')
        # ax.set_xlabel(r'$q_x$')
        if i == 0:
            ax.set_ylabel(r'$q_y$')
            ax.text(-0.7, 0.5, 'Ext. flow data', transform=ax.transAxes,rotation='vertical',horizontalalignment='center',verticalalignment='center')
        # ax.set_title('Pe = %.2f'%Pelist[i])

    # for i in range(len(Pelist)-2):
    for i in range(len(Pelist)):
        ax = axs[3,i]
        cax = ax.pcolormesh(qxlist,qylist,Ie_pred[i],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax.set_aspect('equal')
        ax.set_xlabel(r'$q_x$')
        if i == 0:
            ax.set_ylabel(r'$q_y$')
            ax.text(-0.7, 0.5, 'Ext. flow SIMPLE', transform=ax.transAxes,rotation='vertical',horizontalalignment='center',verticalalignment='center')
        # ax.set_title('Pe = %.2f'%Pelist[i])

    # for i in range(len(Pelist)-2,len(Pelist)):
    #     ax = axs[3,i]
    #     caxlin = ax.pcolormesh(qxlist,qylist,Ie_pred[i],norm=colors.SymLogNorm(linthresh=0.5, linscale=1.0, vmin=np.min(Ie_pred[-1]),vmax=np.max(Ie_pred[-1]), base=10),cmap=mymap,shading='gouraud')
    #     ax.set_aspect('equal')
    #     ax.set_xlabel(r'$q_x$')
    #     if i == 0:
    #         ax.set_ylabel(r'$q_y$')
    #         ax.text(-0.7, 0.5, 'Ext. flow SIMPLE', transform=ax.transAxes,rotation='vertical',horizontalalignment='center',verticalalignment='center')
    #     # ax.set_title('Pe = %.2f'%Pelist[i])

    ax.set_yticks([-0.1,0.0,0.1])
    ax.set_xticks([-0.1,0.0,0.1])

    # cb2 = fig.colorbar(caxlin, ax=axs.ravel().tolist(),aspect=40,pad=-0.07)
    # cb2.ax.set_title(r'$I(q)$')

    cb = fig.colorbar(cax, ax=axs.ravel().tolist(),aspect=40,pad=0.03)
    cb.ax.set_title(r'$I(q)$')

    # plt.tight_layout()
    plt.savefig(open('ss_snaps.png','wb'))
#! /home/cyoung37/anaconda3/bin/python3

# Includes upstream flow history, then truncates to region of interest within cross-flow of FFoRM
# This code computes the phase in the co-rotating frame and rotates the scattering intensity to phase-aligned frame
# Outputs:
# IQa phase-aligned intensity
# Q rotation matrix for vorticity
# phase_t_cort corotating frame phase
# Gdotout lab frame velocity gradient tensor evaluated at even intervals by interpolation (not used later, just included here for completeness)
# Eout lab frame rate of strain tensor evaluated at even intervals 
# QaT_stress_Qa phase-aligned stress

import os
import sys
import math
import time

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
import matplotlib.tri as mtri
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

import scipy
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy import linalg
from scipy import integrate
from scipy.linalg import sqrtm
from scipy.io import savemat
from scipy import special
from scipy import ndimage, misc
import scipy.fftpack

parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--FRR', type=float, default=-0.7) 
parser.add_argument('--Q2', type=float, default=10.0)
parser.add_argument('--strm', type=int, default=4)
parser.add_argument('--avg_win', type=float, default=0.01)
parser.add_argument('--nqi', type=int, default=100)
parser.add_argument('--qlim', type=float, default=0.1)
parser.add_argument('--nbins', type=int, default=200)
parser.add_argument('--ntraj', type=int, default=100000)
parser.add_argument('--load', type=int, default=1)
parser.add_argument('--movie', type=int, default=0)
args = parser.parse_args()

def cyl_pq(qx,qy,qz,R,L,gamma,beta,alpha):

    return (4.0*scipy.special.j1(R*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5))*np.sin(0.5*L*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta)))/(L*R*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta))*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5)))**2

def dQdt(t,Q):
    vort = np.transpose(vfunc(t))
    # vort = vfunc(t)
    Q = np.reshape(Q,(3,3))
    return np.reshape(np.matmul(vort,Q),9)

def smooth(q,qcs,qc):
    return 0.5*np.cos(np.pi*(q - qcs)/(qc - qcs)) + 0.5

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

    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    mymap=matplotlib.cm.get_cmap('jet')

    FRR = args.FRR
    Q2 = args.Q2
    strm = args.strm
    avg_win = args.avg_win
    nqi = args.nqi
    qlim = args.qlim
    nbins = args.nbins
    ntraj = args.ntraj
    load = args.load
    movie = args.movie
    samp_rate = 1
    dt = 0.0005
    L = 1680.0;
    R = 34.0;
    volfrac=0.1 # volume fraction of cylinders
    dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
    bkgd = 0.0001 # background scattering in cm^-1
    gamma = 0.0

    # wdir = 'output/FT%.2f_Pe%.1e/'%(flowType,Pe)
    wdir = 'output/FRR_%.2f/Q2_%.2f/strm_%d/Pq_nt%d/'%(FRR,Q2,strm,ntraj)
    train_out_dir = 'preprocessed/na%d_nq_%d_qlim_%.2f_L%.1f_R%.1f/'%(nbins,nqi,qlim,L,R)
    stress_path = 'preprocessed/stress_L%.1f_R%.1f/'%(L,R)

    if os.path.exists(train_out_dir) == 0:
        os.makedirs(train_out_dir)

    t,stress,Gd,E = pickle.load(open(stress_path+'FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'rb'))
    print(stress.shape)
    data = np.loadtxt('../Gamma_input/FRR_%.2f/Q2_%.2f/strm_%d.txt'%(FRR,Q2,strm),skiprows=2)
    tflow = data[:,0]
    flow_type = data[:,1]
    flow_strength = data[:,2]
    Gdot = data[:,3:]
    Gdot = np.reshape(Gdot,(Gdot.shape[0],3,3))
    Dr = 2.0
    tflow = tflow*Dr
    Gdot = Gdot/Dr
    GdotT = np.transpose(Gdot,axes=[0,2,1])
    vort = 0.5*(Gdot - GdotT)
    vfunc = interp1d(tflow, vort, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    dt = 0.0005
    tend = tflow[-1]
    tmax = int(tend/dt)

    print('No. time steps ',tmax)

    gfunc = interp1d(tflow, Gdot, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

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
    Qmag = np.zeros(nQ)
    xcount = 0
    for Qxi in qxlist:
        ycount = 0
        for Qyi in qylist:
            Qx[xcount + Qxnum*ycount] = Qxi
            Qy[xcount + Qxnum*ycount] = Qyi
            ycount = ycount + 1
        xcount = xcount + 1

    Qmag = np.sqrt(Qx**2 + Qy**2)
    azim = np.arctan2(Qy,Qx)

    samp_rate = 1
    steps_p_win = int(avg_win/(dt*samp_rate))
    print('Time steps per window %d'%steps_p_win)
    n_win = math.ceil(tend/avg_win)
    print('Number of windows %d'%n_win)
    P = np.zeros((n_win,nQ))
    print('P array size %f GB'%(8.0*P.size/1e9))
    snap_count = np.zeros(n_win)

    teval = np.arange(0,n_win)*avg_win
    Q0 = np.array([[1,0,0],[0,1,0],[0,0,0]])
    sol = scipy.integrate.solve_ivp(dQdt,[0,tflow[-1]],Q0.flatten(),t_eval = teval, max_step = 0.01)
    Q = np.transpose(sol.y)
    thetaQ = np.arctan2(Q[:,1],Q[:,0])

    for w in range(1,n_win):
        if abs(thetaQ[w] - thetaQ[w-1]) > (2*np.pi - np.pi/4):
            thetaQ[w] -= 2.0*np.pi*np.round((thetaQ[w] - thetaQ[w-1])/(2.0*np.pi))

    plt.figure(figsize=(4,3),dpi=200)
    plt.plot(teval,Q[:,0],label=r'$Q_{xx}$')
    plt.plot(teval,Q[:,1],label=r'$Q_{xy}$')
    plt.plot(teval,Q[:,3],'--',label=r'$Q_{yx}$')
    plt.plot(teval,Q[:,4],'--',label=r'$Q_{yy}$')
    plt.plot(teval,thetaQ,':',label=r'$\theta$')
    # plt.plot(tflow,np.sin(tflow/2.0),':',label='sint')
    plt.ylabel('Rotation vector components')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # exit()
    plt.savefig(open(train_out_dir+'Q_evo_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))

    Q = np.reshape(Q,(len(Q),3,3))
    QT = np.transpose(Q,axes=[0,2,1])

    Gdotout = gfunc(teval)
    Gdotout = np.reshape(Gdotout,(Gdotout.shape[0],3,3))
    Eout = 0.5*(Gdotout + np.transpose(Gdotout,axes=[0,2,1]))
    Gdotout = np.reshape(Gdotout,(Gdotout.shape[0],9))
    Eout = np.reshape(Eout,(Eout.shape[0],9))

    if load == 0:
        h = np.loadtxt(wdir+'h_na_%d.txt'%(nbins))
        print('PDF input shape')
        print(h.shape)
        h = np.reshape(h,(n_win,nbins,nbins))
        print('PDF new shape')
        print(h.shape)

        start_time = time.time()
        dangle = np.pi/nbins

        cum_weight = np.zeros(n_win)
        for jt in range(nbins):
            dtheta = (jt + 0.5)*dangle
            for jp in range(nbins):
                dphi = (jp + 0.5)*dangle
                Pj = cyl_pq(Qx,Qy,Qz,R,L,gamma,dtheta,dphi)
                for nw in range(n_win):
                    cum_weight[nw] += 2.0*dangle*dangle*np.sin(dtheta)*h[nw][jt][jp]
                    P[nw] += 2.0*dangle*dangle*np.sin(dtheta)*h[nw][jt][jp]*Pj

            print("--- jt %d dtheta %f %s seconds ---" % (jt,dtheta,time.time() - start_time))

        print("--- %s seconds ---" % (time.time() - start_time))
        I = 1.0e8*np.pi*R**2*L*volfrac*dsld**2*P/cum_weight[:,np.newaxis] + bkgd
        P = np.reshape(P,(n_win,Qxnum,Qynum))
        I = np.reshape(I,(n_win,Qxnum,Qynum))
            
        pickle.dump([I,cum_weight],open(wdir+'Iq_%d_%d.p'%(ntraj,nbins),'wb'))

    else:
        I,cum_weight = pickle.load(open(wdir+'Iq_%d_%d.p'%(ntraj,nbins),'rb'))

    kernel_size = np.max((1,int(len(I)/20)))
    # kernel_size = 1
    I = rolling_mean_along_axis(I,kernel_size,0)
    stress = rolling_mean_along_axis(stress,kernel_size,0)

    print(I.shape)
    Qmag = np.reshape(Qmag,(Qxnum,Qynum))
    azim = np.reshape(azim,(Qxnum,Qynum))

    I = np.where(Qmag[np.newaxis,:,:] < qlim, I, 0.0)
    # Optional gating function for smooth transition to zero at qlim, see smooth func at top of code
    # qcs = 0.08
    # I = np.where(np.logical_and(np.greater_equal(Qmag[np.newaxis,:,:],qcs),np.less_equal(Qmag[np.newaxis,:,:],qlim)), I*smooth(Qmag,qcs,qlim), I)
    IQ = np.zeros(I.shape)
    for i in range(n_win):
        IQ[i] = ndimage.rotate(I[i],thetaQ[i]*180.0/np.pi,reshape=False)
        IQ[i] = np.where(Qmag[np.newaxis,:,:] < 0.1, IQ[i], 0.0)

    qmax = 0.05
    qmin = 0.03
    Imask = np.where(np.logical_and(np.greater_equal(Qmag[np.newaxis,:,:],qmin),np.less_equal(Qmag[np.newaxis,:,:],qmax)), I, 0.0)

    nazbin = 128
    azbinsize = np.pi/nazbin
    avg_theta = np.zeros((n_win,nazbin))
    Ibin = np.zeros((n_win,nazbin,Qxnum,Qynum))
    for i in range(nazbin):
        imin = i*azbinsize
        imax = (i+1)*azbinsize
        for j in range(n_win):
            Ibin[j,i] = np.where(np.logical_and(np.greater_equal(azim[np.newaxis,:,:],imin),np.less_equal(azim[np.newaxis,:,:],imax)), Imask[j], 0.0)
            temp = Ibin[j,i]
            temp = temp[np.nonzero(temp > 0.0)]
            avg_theta[j,i] = np.mean(temp)

    thetarange = np.arange(0,nazbin)*azbinsize

    phase_t = np.zeros(n_win)
    for i in range(n_win):
        ft_Iq = np.fft.fft(avg_theta[i])
        # phase_t[i] = np.angle(ft_Iq)[1] - np.pi/2
        phase_t[i] = np.angle(ft_Iq)[1]

    phase_t_nonwrap = np.copy(phase_t)
    for w in range(1,n_win):
        if abs(phase_t[w] - phase_t[w-1]) > (np.pi - nazbin/2*azbinsize):
            phase_t[w] -= np.pi*np.sign(phase_t[w] - phase_t[w-1])*np.ceil(np.abs((phase_t[w] - phase_t[w-1]))/np.pi)

    phase_t_corot = 2.0*(phase_t/2.0 + thetaQ)

    # fig, ax = plt.figure(figsize=(4,3),dpi=200)
    fig = plt.figure(figsize=(4,3),dpi=200)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax2.plot(teval,-phase_t_corot/2,'--',label=r'$\alpha$')
    ax2.plot(teval,-phase_t/2,'--',label=r'$\alpha_{lab}$')
    ax.plot(teval,Q[:,0],':',label=r'$Q_xx$')
    ax.plot(teval,Q[:,1],'-.',label=r'$Q_xy$')
    ax.set_ylabel(r'$Q_{xx},Q_{yy}$')
    # ax2.set_ylabel(r'$Q_{xx},Q_{yy}$')
    ax2.set_ylabel(r'$\alpha, \alpha_{lab}$')
    plt.xlabel(r'$\mathcal{T}$')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # exit()
    plt.savefig(open(train_out_dir+'phasel2c_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))
    plt.close()

    plt.figure(figsize=(4,3),dpi=200)
    # plt.plot(teval,thetaQ*180.0/np.pi,label=r'$\theta$')
    plt.plot(teval,-phase_t*90.0/np.pi,'--',label=r'$\alpha_{lab}$')
    # plt.plot(teval,-phase_t*90.0/np.pi - thetaQ*180.0/np.pi,':',label=r'$\alpha_{corot}$')
    # plt.plot(teval,-phase_t_corot*90.0/np.pi,'-.',label=r'$\alpha_{corot,out}$')
    # plt.plot(teval,-phase_t_lab*90.0/np.pi,'x',markersize=0.5,label=r'$\alpha_{lab}$')
    # plt.plot(tflow,np.sin(tflow/2.0),':',label='sint')
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Time '+r'$\tilde{t} = tD_r$')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # exit()
    plt.savefig(open(train_out_dir+'theta_phase_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))
    plt.close()

    IQa = np.zeros(I.shape)
    QaT_stress_Qa = np.zeros((n_win,3,3))
    back = np.zeros((n_win,3,3))
    for i in range(n_win):
        IQa[i] = ndimage.rotate(I[i],-phase_t[i]*90.0/np.pi,reshape=False)
        IQa[i] = np.where(Qmag[np.newaxis,:,:] < qlim, IQa[i], 0.0)
        phaseQ = -phase_t[i]/2.0
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,1.0]])
        QaT_stress_Qa[i] = np.transpose(RQ)@stress[i]@RQ
        back[i] = RQ@QaT_stress_Qa[i]@np.transpose(RQ)

    QaT_stress_Qa = np.reshape(QaT_stress_Qa,(QaT_stress_Qa.shape[0],9))

    fig = plt.figure(figsize=(8,6),dpi=200)
    ax = plt.subplot(221)
    plt.plot(teval,stress[:,0,0],'-',label='Lab')
    plt.plot(teval,QaT_stress_Qa[:,0],'--',label='phase-aligned')
    plt.plot(teval,back[:,0,0],':',label='QQTsQQT')
    plt.legend()
    plt.ylabel(r'$QTsQ_{xx}$')

    ax = plt.subplot(222)
    plt.plot(teval,stress[:,0,1],'-',label='Lab')
    plt.plot(teval,QaT_stress_Qa[:,1],'--',label='Lab')
    plt.plot(teval,back[:,0,1],':',label='QQTsQQT')
    plt.ylabel(r'$QTsQ_{xy}$')

    ax = plt.subplot(223)
    plt.plot(teval,stress[:,1,0],'-',label='Lab')
    plt.plot(teval,QaT_stress_Qa[:,3],'--',label='Lab')
    plt.plot(teval,back[:,1,0],':',label='QQTsQQT')
    plt.ylabel(r'$QTsQ_{yx}$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(224)
    plt.plot(teval,stress[:,1,1],'-',label='Lab')
    plt.plot(teval,QaT_stress_Qa[:,4],'--',label='Lab')
    plt.plot(teval,back[:,1,1],':',label='QQTsQQT')
    plt.ylabel(r'$QTsQ_{yy}$')
    plt.xlabel(r'$t$')

    plt.tight_layout()
    plt.savefig(open(train_out_dir+'stress_FRR_%.2f_Q2_%.2f_strm_%d.png'%(FRR,Q2,strm),'wb'))
    # plt.show()

    if movie == 1:
        t = np.linspace(0,10,n_win)
        # mymap=matplotlib.cm.get_cmap('inferno')
        mymap=matplotlib.cm.get_cmap('jet')
        vmin = 1e-3
        vmax = 600

        fig = plt.figure(figsize=(10,3),dpi=200)
        ax = fig.add_subplot(131)

        cax = ax.pcolormesh(qxlist,qylist,I[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax.set_aspect('equal')
        ax.set_xlabel(r'$q_x [\AA^{-1}]$')
        ax.set_ylabel(r'$q_y [\AA^{-1}]$')

        ax2 = fig.add_subplot(132)
        cax2 = ax2.pcolormesh(qxlist,qylist,IQ[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax2.set_aspect('equal')
        ax2.set_xlabel(r'$q_x [\AA^{-1}]$')
        ax2.yaxis.set_ticklabels([])

        ax3 = fig.add_subplot(133)
        cax3 = ax3.pcolormesh(qxlist,qylist,IQa[0],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
        ax3.set_aspect('equal')
        ax3.yaxis.set_ticklabels([])
        ax3.set_xlabel(r'$q_x [\AA^{-1}]$')
        cb3 = fig.colorbar(cax3,ax=ax3)
        cb3.ax.set_title(r'$I(q)$')

        plt.tight_layout()
        # plt.show()

        def animate(i):
           # cax.set_array(I[i])
           cax.set_array(I[i])
           ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           cax2.set_array(IQ[i])
           # ax2.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           ax2.set_title(r'$\theta = %.2f (degrees)$'%(thetaQ[i]*180.0/np.pi))
           cax3.set_array(IQa[i])
           # ax3.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
           ax3.set_title(r'$\alpha_{lab} = %.2f, \alpha_c = %.2f$'%(-phase_t[i]*90.0/np.pi,-phase_t_corot[i]*90.0/np.pi))

        # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
        # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=np.max((n_win,20)))
        t0 = pickle.load(open('../Gamma_input_upstream/FRR_%.2f/Q2_%.2f/tcut_%d.p'%(FRR,Q2,strm),'rb'))
        t0 = t0*Dr
        print(t0)
        ind0 = np.argwhere(teval > t0)
        # Find first index (earliest time)
        ind0 = np.min(ind0)
        print(ind0,t0)
        frames = np.arange(ind0,n_win)
        anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=frames)
        # anim.save(train_out_dor+'IqR_%d_%d.mp4'%(ntraj,nbins))
        anim.save(train_out_dir+'Qphase_FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))

    # IQa = np.where(Qmag[np.newaxis,:,:] > qlim, 0.0, IQa)
    IQa = np.reshape(IQa,(n_win,nqi*nqi))
    ind = np.nonzero(IQa[0])
    print('Number of nonzero indices %d'%len(ind[0]))
    # print(ind[0])
    IQa = IQa[np.nonzero(IQa)]
    lenIQa = int(len(IQa)/len(ind[0]))
    IQa = np.reshape(IQa,(lenIQa,len(ind[0])))
    print('Output rotated pattern shape ',IQa.shape)
    # print(ind)
    # zero_ind = np.delete(np.arange(0,nqi*nqi),ind)
    # print(zero_ind)
    Q = Q[:lenIQa]
    phase_t_corot = phase_t_corot[:lenIQa]
    Gdotout = Gdotout[:lenIQa]
    Eout = Eout[:lenIQa]
    QaT_stress_Qa = QaT_stress_Qa[:lenIQa]
    # print(IQa.shape,Gdotout.shape)
    print('phase shape, Q shape, Gdshape, Eshape, stress shape',phase_t_corot.shape,Q.shape,Gdotout.shape,Eout.shape,QaT_stress_Qa.shape)

    # tcut = np.loadtxt('../Gamma_input_upstream/FRR_%.2f/Q2_%.2f/strm_%d.txt'%(FRR,Q2,strm),skiprows=2)
    t0 = pickle.load(open('../Gamma_input_upstream/FRR_%.2f/Q2_%.2f/tcut_%d.p'%(FRR,Q2,strm),'rb'))
    t0 = t0*Dr
    print(t0)
    ind0 = np.argwhere(teval > t0)
    # Find first index (earliest time)
    ind0 = np.min(ind0)
    print(ind0,t0)

    IQa = IQa[ind0:]
    Q = Q[ind0:]
    phase_t_corot = phase_t_corot[ind0:]
    Gdotout = Gdotout[ind0:]
    Eout = Eout[ind0:]
    QaT_stress_Qa = QaT_stress_Qa[ind0:]
    print('IQa shape,phase shape, Q shape, Gdshape, Eshape, stress shape',IQa.shape,phase_t_corot.shape,Q.shape,Gdotout.shape,Eout.shape,QaT_stress_Qa.shape)

    pickle.dump([IQa,Q,phase_t_corot,Gdotout,Eout,QaT_stress_Qa],open(train_out_dir+'IQphase_FRR_%.2f_Q2_%.2f_strm_%d.p'%(FRR,Q2,strm),'wb'))
    exit()

    if movie == 1:
        Gdot = gfunc(teval)
        GdotT = np.transpose(Gdot,axes=[0,2,1])
        E = 0.5*(Gdot + GdotT)
        # print(Gdot.shape)
        # Gdot_corot = np.zeros(Gdot.shape)
        QTEQ = np.zeros(E.shape)
        aTEa = np.zeros(E.shape)
        for i in range(E.shape[0]):
            # QTEQ[i] = QT[i]@E[i]@Q[i]
            QTEQ[i] = Q[i]@E[i]@QT[i]
            thetaQi = np.arctan2(Q[i,0,1],Q[i,0,0])
            # phaseQ = -phase_t[i]/2.0 + thetaQi
            phaseQ = -phase_t[i]/2.0
            RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,0.0]])
            # print(phaseQ.shape,RQ.shape)
            # exit()
            aTEa[i] = np.transpose(RQ)@E[i]@RQ
            # aTEa[i] = np.transpose(RQ)@Gd[i]@RQ

        # Visualize on grid
        w = 3
        ngrid = 25
        grid = np.linspace(-w,w,ngrid)
        Y, X = np.mgrid[-w:w:np.complex(ngrid), -w:w:np.complex(ngrid)]
        # Y, X = np.meshgrid(grid, grid)
        U = np.zeros((len(Gdot),ngrid,ngrid))
        V = np.zeros((len(Gdot),ngrid,ngrid))
        UE = np.zeros((len(Gdot),ngrid,ngrid))
        VE = np.zeros((len(Gdot),ngrid,ngrid))
        UEc = np.zeros((len(Gdot),ngrid,ngrid))
        VEc = np.zeros((len(Gdot),ngrid,ngrid))
        for i in range(len(U)):
            U[i] = Gdot[i,0,0]*X + Gdot[i,0,1]*Y
            V[i] = Gdot[i,1,0]*X + Gdot[i,1,1]*Y
            UE[i] = aTEa[i,0,0]*X + aTEa[i,0,1]*Y
            VE[i] = aTEa[i,1,0]*X + aTEa[i,1,1]*Y
            UEc[i] = QTEQ[i,0,0]*X + QTEQ[i,0,1]*Y
            VEc[i] = QTEQ[i,1,0]*X + QTEQ[i,1,1]*Y

        speed = np.sqrt(U**2 + V**2)
        Enorm = np.sqrt(UE**2 + VE**2)
        Ecnorm = np.sqrt(UEc**2 + VEc**2)
        print(np.min(speed),np.max(speed))
        print(speed.shape)

        cmin = 0
        cmax = np.max([speed])
        print(cmin,cmax)

        fig = plt.figure(figsize=(10.5,3),dpi=200)

        ax1 = fig.add_subplot(131)
        cax1 = ax1.pcolormesh(X,Y,speed[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
        cb = fig.colorbar(cax1,ax=ax1)
        # cb.ax.set_title(r'$|\nabla \boldsymbol{v}|$')
        cb.ax.set_title(r'$|\nabla v|$')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')
        ax1.set_xlim(-3,3)
        ax1.set_ylim(-3,3)
        ax1.set_aspect('equal')

        ax2 = fig.add_subplot(132)
        cax2 = ax2.pcolormesh(X,Y,Enorm[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
        cb = fig.colorbar(cax2,ax=ax2)
        cb.ax.set_title(r'$|E_{\alpha}|$')
        ax2.set_xlabel(r'$x$')
        ax2.set_xlim(-3,3)
        ax2.set_ylim(-3,3)
        ax2.set_aspect('equal')

        ax3 = fig.add_subplot(133)
        cax3 = ax3.pcolormesh(X,Y,Ecnorm[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
        cb3 = fig.colorbar(cax3,ax=ax3)
        cb3.ax.set_title(r'$|E_c|$')
        ax3.set_xlabel(r'$x$')
        ax3.set_xlim(-3,3)
        ax3.set_ylim(-3,3)
        ax3.set_aspect('equal')

        ax1.streamplot(X,Y,U[0],V[0],color='white',linewidth=1.0)
        ax2.streamplot(X,Y,UE[0],VE[0],color='white',linewidth=1.0)
        ax3.streamplot(X,Y,UEc[0],VEc[0],color='white',linewidth=1.0)

        plt.tight_layout()
        # plt.show()
        # exit()

        def animate(i):

            # ax3.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
            ax1.set_title(r'$\alpha_{lab} = %.2f$'%(-phase_t[i]*90.0/np.pi))
            ax2.set_title(r'$\alpha_{c} + \theta_Q = %.2f$'%(-phase_t_corot[i]*90.0/np.pi + thetaQ[i]*180.0/np.pi))
            ax3.set_title(r'$\theta_Q = %.2f$'%(thetaQ[i]*180.0/np.pi))
            # ax3.set_title(r'$|\textbf{E}_1 - \textbf{E}_2| = %.2e$'%np.mean(Enorm - Ecnorm)**2)
            # print(ax.collections)
            # ax.collections = [] # clear lines streamplot
            # print(ax.collections[0],ax.collections[1])
            # exit()
            ax1.collections = [ax1.collections[0]]
            ax2.collections = [ax2.collections[0]]
            ax3.collections = [ax3.collections[0]]

            # Clear arrowheads streamplot.
            for artist in ax1.get_children():
                if isinstance(artist, FancyArrowPatch):
                    artist.remove()
            for artist in ax2.get_children():
                if isinstance(artist, FancyArrowPatch):
                    artist.remove()
            for artist in ax3.get_children():
                if isinstance(artist, FancyArrowPatch):
                    artist.remove()

            # stream = ax.streamplot(X,Y,U[i],V[i],color='black')
            ax1.streamplot(X,Y,U[i],V[i],color='white',linewidth=1)
            ax2.streamplot(X,Y,UE[i],VE[i],color='white',linewidth=1)
            ax3.streamplot(X,Y,UEc[i],VEc[i],color='white',linewidth=1)
            cax1.set_array(speed[i])
            cax2.set_array(Enorm[i])
            cax2.set_array(Ecnorm[i])
            print(i)
            # return stream

        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(Gdot), interval=100, blit=False, repeat=False)
        anim.save(train_out_dir+'flow_rotphase_FRR_%.2f_Q2_%.2f_strm_%d.mp4'%(FRR,Q2,strm))

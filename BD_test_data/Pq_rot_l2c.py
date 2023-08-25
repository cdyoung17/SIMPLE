#! /home/cyoung37/anaconda3/bin/python3

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
from scipy import integrate
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.io import savemat
from scipy import special
from scipy import ndimage, misc
import scipy.fftpack

parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--flowType', type=float, default=1.0) 
parser.add_argument('--Pe', type=float, default=10.0)
parser.add_argument('--avg_win', type=float, default=0.01)
parser.add_argument('--nqi', type=int, default=100)
parser.add_argument('--qlim', type=float, default=0.1)
parser.add_argument('--nbins', type=int, default=200)
parser.add_argument('--ntraj', type=int, default=100000)
parser.add_argument('--load', type=int, default=1)
parser.add_argument('--tcut', type=int, default=200)
args = parser.parse_args()

def cyl_pq(qx,qy,qz,R,L,gamma,beta,alpha):

    return (4.0*scipy.special.j1(R*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5))*np.sin(0.5*L*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta)))/(L*R*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta))*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5)))**2

def dmdt(t,m):

    m = m/np.linalg.norm(m)
    vort = vfunc(t)
    return np.matmul(vort,m)

def dQdt(t,Q):
    vort = np.transpose(vfunc(t))
    # vort = vfunc(t)
    Q = np.reshape(Q,(3,3))
    return np.reshape(np.matmul(vort,Q),9)

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

    flowType = args.flowType
    Pe = args.Pe
    avg_win = args.avg_win
    nqi = args.nqi
    qlim = args.qlim
    nbins = args.nbins
    ntraj = args.ntraj
    load = args.load
    wdir = 'output/FT%.2f_Pe%.1e/'%(flowType,Pe)

    data = np.loadtxt('Gamma_input/FT%.2f_Pe%.1e.txt'%(flowType,Pe),skiprows=2)
    tcut = args.tcut
    # tcut = len(data)
    data = data[:tcut]
    tflow = data[:,0]
    flow_type = data[:,1]
    flow_strength = data[:,2]
    Gdot = data[:,3:]
    Gdot = np.reshape(Gdot,(Gdot.shape[0],3,3))
    Dr = 2.0
    tflow = tflow*Dr
    Gdot = Gdot/Dr
    GdotT = np.transpose(Gdot,axes=[0,2,1])
    E = 0.5*(Gdot + GdotT)
    vort = 0.5*(Gdot - GdotT)
    dt = 0.0005
    tend = tflow[-1]
    tmax = int(tend/dt)

    samp_rate = 1
    steps_p_win = int(avg_win/(dt*samp_rate))
    print('Time steps per window %d'%steps_p_win)
    n_win = math.ceil(tend/avg_win)
    print('tend %f'%tend)
    print('Number of windows %d'%n_win)

    gfunc = interp1d(tflow, Gdot, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    vfunc = interp1d(tflow, vort, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Efunc = interp1d(tflow, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    m10 = np.array([1.0,0.0,0.0])
    m20 = np.array([0.0,1.0,0.0])

    teval = np.arange(0,n_win)*avg_win
    sol = scipy.integrate.solve_ivp(dmdt,[0,tflow[-1]],m10,t_eval = teval, max_step = 0.01)
    m1 = np.transpose(sol.y)

    sol = scipy.integrate.solve_ivp(dmdt,[0,tflow[-1]],m20,t_eval = teval, max_step = 0.01)
    m2 = np.transpose(sol.y)

    m1mag = np.linalg.norm(m1,axis=1)
    m10dotm1 = np.einsum('j,ij->i',m1[0],m1)

    thetacos = np.arccos(m10dotm1)
    # theta = np.arctan2(m1[:,0],m1[:,1]) - np.pi/2
    theta = np.arctan2(m1[:,1],m1[:,0])

    # for i in range(1,len(theta)):
    #     if abs(theta[i] - theta[i-1]) > np.pi:
    #         # print(i,tflow[i],theta[i],theta[i-1],np.round((theta[i] - theta[i-1])/(2*np.pi)))
    #         # theta[i] += 2*np.pi
    #         theta[i] -= 2*np.pi*np.round((theta[i] - theta[i-1])/(2*np.pi))

    plt.figure(figsize=(4,3),dpi=200)
    plt.plot(teval,m1[:,0],label=r'$m_{1x}$')
    plt.plot(teval,m1[:,1],label=r'$m_{1y}$')
    plt.plot(teval,m2[:,0],'--',label=r'$m_{2x}$')
    plt.plot(teval,m2[:,1],'--',label=r'$m_{2y}$')
    plt.plot(teval,theta,label=r'$\theta$')
    # plt.plot(tflow,thetacos,'--',label=r'$\theta_{cos}$')
    # plt.plot(tflow,m1[:,2])
    plt.ylabel('Basis vector components')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(open(wdir+'basis_evolution.png','wb'))
    # exit()

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
    plt.plot(teval,thetaQ,label=r'$\theta$')
    # plt.plot(tflow,np.sin(tflow/2.0),':',label='sint')
    plt.ylabel('Rotation vector components')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # exit()
    plt.savefig(open(wdir+'Q_evolution.png','wb'))

    Q = np.reshape(Q,(len(Q),3,3))
    QT = np.transpose(Q,axes=[0,2,1])
    Qfunc = interp1d(teval, Q, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")

    print('No. time steps ',tmax)

    dt = 0.0005

    # L = 9200;
    # R = 33;
    L = 1680.0;
    R = 34.0;
    volfrac=0.1 # volume fraction of cylinders
    dsld = (6.33E-6)-(3.13E-6) # scattering length density difference between cylinder and solvent (in Angstroms^2)
    bkgd = 0.0001 # background scattering in cm^-1
    gamma = 0.0

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
            # Qmag[xcount + Qxnum*ycount] = np.sqrt(Qxi**2 + Qyi**2)
            # Qx[Qxnum*xcount + ycount] = Qxi
            # Qy[Qxnum*xcount + ycount] = Qyi
            ycount = ycount + 1
        xcount = xcount + 1

    Qmag = np.sqrt(Qx**2 + Qy**2)
    azim = np.arctan2(Qy,Qx)
    P = np.zeros((n_win,nQ))
    print('P array size %f GB'%(8.0*P.size/1e9))
    snap_count = np.zeros(n_win)

    if load == 0:
        # h = np.loadtxt(wdir+'h_na_%d_nq_%d_qlim_%.2f.txt'%(nbins,Qxnum,Qxmax))
        h = np.loadtxt(wdir+'h_na_%d_%d.txt'%(ntraj,nbins))
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
        I = I[:n_win]
        cum_weight = cum_weight[:n_win]

    print('I shape',I.shape)
    print('Number of windows',n_win)
    Qmag = np.reshape(Qmag,(Qxnum,Qynum))
    azim = np.reshape(azim,(Qxnum,Qynum))

    # kernel_size = np.max((1,int(len(I)/20)))
    kernel_size = 1
    I = rolling_mean_along_axis(I,kernel_size,0)

    I = np.where(Qmag[np.newaxis,:,:] < qlim, I, 0.0)
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
    # i = 0
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

    for w in range(1,n_win):
        if abs(phase_t[w] - phase_t[w-1]) > (np.pi - nazbin/2*azbinsize):
            phase_t[w] -= np.pi*np.round((phase_t[w] - phase_t[w-1])/np.pi)

    phase_t_corot = 2.0*(phase_t/2.0 + thetaQ)

    win1 = 1
    win2 = 20
    win3 = 40
    # plt.figure(figsize=(4,3),dpi=200)
    fig, ax = plt.subplots(figsize=(4,3.37),dpi=200)
    ax2 = ax.twinx()
    # plt.plot(teval,thetaQ*180.0/np.pi,label=r'$\theta$')
    ax.plot(teval*Pe,Q[:,0,0],label=r'$Q_{xx}$')
    ax.plot(teval*Pe,Q[:,0,1],'--',label=r'$Q_{xy}$')
    ax2.plot(teval*Pe,-phase_t_corot/2,'-.',color='red',label=r'$\alpha$')
    ax2.plot(teval*Pe,-phase_t/2,':',color='black',label=r'$\alpha_{lab}$')
    # ax.plot(teval,Q[:,0,0],label=r'$Q_{xx}$')
    # ax.plot(teval,Q[:,0,1],'--',label=r'$Q_{xy}$')
    # ax2.plot(teval,-phase_t_corot/2,'-.',color='red',label=r'$\alpha$')
    # ax2.plot(teval,-phase_t/2,':',color='black',label=r'$\alpha_{lab}$')
    # plt.plot(tflow,np.sin(tflow/2.0),':',label='sint')
    # plt.ylabel('Co-rotating angle, phase angle (degrees)')
    ax.set_ylabel(r'$Q_{xx},Q_{xy}$')
    # ax.set_ylabel(r'$\textrm{Q}_{xx},\textrm{Q}_{yy}$')
    ax2.set_ylabel(r'$\alpha,\alpha_{lab}$ (radians)')
    # xticks = np.pi/4*np.arange(-3,2)
    xticks = np.pi/4*np.arange(0,6)
    labels = ['0',r'$\pi/4$',r'$\pi/2$',r'$3 \pi/4$',r'$\pi$',r'$5\pi/4$']
    # labels = [r'$-3\pi/4$',r'-$\pi/2$',r'$-\pi/4$',r'$0$',r'$\pi/4$'])
    plt.yticks(xticks,labels)
    ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_xlabel(r'$\gamma = \dot{\gamma} \mathcal{T}$')
    # ax.set_xlabel(r'$\mathcal{T}$')
    ax.axvline(x=win1*Pe*avg_win,ls=':',lw=1.0,color='gray')
    ax.axvline(x=win2*Pe*avg_win,ls=':',lw=1.0,color='gray')
    ax.axvline(x=win3*Pe*avg_win,ls=':',lw=1.0,color='gray')
    # plt.legend()
    fig.legend(loc="upper right", bbox_to_anchor=(1.0,0.82), bbox_transform=ax.transAxes)
    fig.tight_layout()
    # plt.show()
    # exit()
    ax.text(-2,1,'g)')
    plt.savefig(open(wdir+'theta_phase_%d.png'%len(Q),'wb'))

    t,stress,Gd,E = pickle.load(open(wdir+'t_stress_grad_E_%d.p'%(ntraj),'rb'))
    # stress = stress[:tcut]

    IQa = np.zeros(IQ.shape)
    QaT_stress_Qa = np.zeros((n_win,3,3))
    back = np.zeros((n_win,3,3))
    for i in range(n_win):
        IQa[i] = ndimage.rotate(I[i],-phase_t[i]*90.0/np.pi,reshape=False)
        IQa[i] = np.where(Qmag[np.newaxis,:,:] < qlim, IQa[i], 0.0)
        phaseQ = -phase_t[i]/2.0
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,1.0]])
        # QaT_stress_Qa[i] = np.transpose(RQ)@stress[i]@RQ
        # back[i] = RQ@QaT_stress_Qa[i]@np.transpose(RQ)

    # QaT_stress_Qa = np.reshape(QaT_stress_Qa,(QaT_stress_Qa.shape[0],9))

    # fig = plt.figure(figsize=(8,6),dpi=200)
    # ax = plt.subplot(221)
    # plt.plot(teval,stress[:,0,0],'-',label='Lab')
    # plt.plot(teval,QaT_stress_Qa[:,0],'--',label='phase-aligned')
    # plt.plot(teval,back[:,0,0],':',label='QQTsQQT')
    # plt.legend()
    # plt.ylabel(r'$QTsQ_{xx}$')
    # # plt.xlabel(r'$t$')

    # ax = plt.subplot(222)
    # plt.plot(teval,stress[:,0,1],'-',label='Lab')
    # plt.plot(teval,QaT_stress_Qa[:,1],'--',label='Lab')
    # plt.plot(teval,back[:,0,1],':',label='QQTsQQT')
    # plt.ylabel(r'$QTsQ_{xy}$')

    # ax = plt.subplot(223)
    # plt.plot(teval,stress[:,1,1],'-',label='Lab')
    # plt.plot(teval,QaT_stress_Qa[:,4],'--',label='Lab')
    # plt.plot(teval,back[:,1,1],':',label='QQTsQQT')
    # plt.ylabel(r'$QTsQ_{yy}$')
    # plt.xlabel(r'$t$')

    # ax = plt.subplot(224)
    # plt.plot(teval,stress[:,2,2],'-',label='Lab')
    # plt.plot(teval,QaT_stress_Qa[:,8],'--',label='Lab')
    # plt.plot(teval,back[:,2,2],':',label='QQTsQQT')
    # plt.ylabel(r'$QTsQ_{zz}$')
    # plt.xlabel(r'$t$')

    # plt.tight_layout()
    # plt.savefig(open(wdir+'stress_%d.png'%(ntraj),'wb'))

    IQa_out = np.reshape(IQa,(n_win,nqi*nqi))
    ind = np.nonzero(IQa[0])
    print('Number of nonzero indices %d'%len(ind[0]))
    # print(ind[0])
    IQa_out = IQa_out[np.nonzero(IQa_out)]
    lenIQa = int(len(IQa_out)/len(ind[0]))
    IQa_out = np.reshape(IQa_out,(lenIQa,len(ind[0])))
    print('Output rotated pattern shape ',IQa_out.shape)
    Q = Q[:lenIQa]
    phase_t_corot = phase_t_corot[:lenIQa]

    Gdotout = gfunc(teval)
    Gdotout = np.reshape(Gdotout,(Gdotout.shape[0],3,3))
    Eout = 0.5*(Gdotout + np.transpose(Gdotout,axes=[0,2,1]))
    Gdotout = np.reshape(Gdotout,(Gdotout.shape[0],9))
    Eout = np.reshape(Eout,(Eout.shape[0],9))
    Gdotout = Gdotout[:lenIQa]
    Eout = Eout[:lenIQa]

    # print('phase shape, Q shape, Gdshape, Eshape, stress shape',phase_t_corot.shape,Q.shape,Gdotout.shape,Eout.shape,QaT_stress_Qa.shape)

    # pickle.dump([IQa_out,Q,phase_t_corot,Gdotout,Eout,QaT_stress_Qa],open(wdir+'l2c_IQphase.p','wb'))
    # exit()

    t = np.linspace(0,10,n_win)
    mymap=matplotlib.cm.get_cmap('jet')
    vmin = 0.01
    vmax = 600

    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(7,3.5),sharex=True,sharey=True,dpi=300)
    # fig,axs = plt.subplots(nrows=3,ncols=2,figsize=(4.75,5.5),sharex=True,sharey=True,dpi=300)

    # window = 1
    # ax1 = plt.subplot(321)
    ax1 = axs[0,0]
    # cax1 = ax1.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    cax1 = ax1.pcolormesh(qxlist,qylist,I[win1],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax1 = ax1.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax1.set_aspect('equal')
    # ax.set_xlabel(r'$q_x$')
    ax1.set_ylabel(r'$q_y$')
    ax1.set_title(r'$\gamma = %.1f$'%(avg_win*Pe*win1))
    # ax1.set_title('True Data')
    ax1.set_yticks([-0.1,0.0,0.1])
    ax1.text(-0.23,-0.09,'Laboratory frame',rotation='vertical')
    ax1.text(-0.18,0.095,'a)')

    # window = 20
    # ax2 = plt.subplot(322)
    ax2 = axs[0,1]
    cax2 = ax2.pcolormesh(qxlist,qylist,I[win2],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax2.set_aspect('equal')
    # ax.set_xlabel(r'$q_x$')
    # ax.set_ylabel(r'$q_y$')
    ax2.set_title(r'$\gamma = %.1f$'%(avg_win*Pe*win2))
    # ax2.set_title('Prediction')
    # ax2.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # ax2.set_yticklabels(['','','','',''])
    # cb2 = fig.colorbar(cax2,ax=ax2)
    # cb2.ax.set_title(r'$I_R(q)$')
    ax2.text(-0.14,0.095,'b)')

    # window = 50
    # ax3 = plt.subplot(323)
    ax3 = axs[0,2]
    # cax3 = ax3.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    cax3 = ax3.pcolormesh(qxlist,qylist,I[win3],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax3 = ax3.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax3.set_aspect('equal')
    # ax3.set_ylabel(r'$q_y$')
    ax3.set_title(r'$\gamma = %.1f$'%(avg_win*Pe*win3))
    # ax3.set_title('Data '+r'$\mathcal{T} = %.2f$'%(dt*window)))
    # ax3.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    ax3.text(-0.14,0.095,'c)')

    # window = 5
    # ax4 = plt.subplot(324)
    ax4 = axs[1,0]
    cax4 = ax4.pcolormesh(qxlist,qylist,IQ[win1],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax4.set_aspect('equal')
    # ax4.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # ax4.set_yticklabels(['','','','',''])
    # ax2.set_title(r'$\gamma = %.2f$'%(dt*Pe*window))
    ax4.set_xlabel(r'$q_x$')
    ax4.set_ylabel(r'$q_y$')
    ax4.text(-0.23,-0.09,'Corotating frame',rotation='vertical')
    ax4.text(-0.18,0.095,'d)')

    # window = 20
    # ax5 = plt.subplot(325)
    ax5 = axs[1,1]
    # cax5 = ax5.pcolormesh(qxlist,qylist,Irot[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    cax5 = ax5.pcolormesh(qxlist,qylist,IQ[win2],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    # cax5 = ax5.pcolormesh(qxlist,qylist,Itilf[window],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax5.set_aspect('equal')
    ax5.set_xlabel(r'$q_x$')
    # ax5.set_ylabel(r'$q_y$')
    # ax5.set_title('Model lab frame')
    # ax5.set_title('Data '+r'$\mathcal{T} = %.2f$'%(dt*window)))
    # ax5.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    ax5.text(-0.14,0.095,'e)')

    # window = 50
    # ax6 = plt.subplot(326)
    ax6 = axs[1,2]
    cax6 = ax6.pcolormesh(qxlist,qylist,IQ[win3],norm=colors.LogNorm(vmin=vmin),cmap=mymap,shading='gouraud')
    ax6.set_aspect('equal')
    ax6.set_xlabel(r'$q_x$')
    # ax6.set_yticks([-0.1,-0.05,0.0,0.05,0.1])
    # ax6.set_yticklabels(['','','','',''])
    # ax6.set_title('NODE '+r'$\mathcal{T} = %.2f$'%(dt*window))
    ax6.text(-0.14,0.095,'f)')

    plt.subplots_adjust(hspace=0.01,wspace=0.3)
    cb = fig.colorbar(cax6, ax=axs.ravel().tolist(),aspect=40)
    cb.ax.set_title(r'$I(q)$')

    # plt.tight_layout()
    plt.savefig(open(wdir+'snaps_lab_corot.png','wb'))
    # exit()

    fig = plt.figure(figsize=(10.5,3),dpi=300)
    ax = fig.add_subplot(131)

    window = 0
    cax = ax.pcolormesh(qxlist,qylist,I[window],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$q_x \ [\AA^{-1}]$')
    ax.set_ylabel(r'$q_y \ [\AA^{-1}]$')
    ax.set_title('Lab frame')
    # cb = fig.colorbar(cax,ax=ax)
    # cb.ax.set_title(r'$I(q)$')

    ax2 = fig.add_subplot(132)
    cax2 = ax2.pcolormesh(qxlist,qylist,IQ[window],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$q_x \ [\AA^{-1}]$')
    ax2.set_title('Co-rotating frame')
    # ax2.set_ylabel(r'$q_y [\AA^{-1}]$')
    # cb2 = fig.colorbar(cax2,ax=ax2)
    # cb2.ax.set_title(r'$I(q)$')

    ax3 = fig.add_subplot(133)
    cax3 = ax3.pcolormesh(qxlist,qylist,IQa[window],norm=colors.LogNorm(vmin=vmin,vmax=vmax),cmap=mymap,shading='gouraud')
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'$q_x \ [\AA^{-1}]$')
    ax3.set_title('Phase-aligned')
    # ax3.set_ylabel(r'$q_y [\AA^{-1}]$')
    cb3 = fig.colorbar(cax3,ax=ax3)
    cb3.ax.set_title(r'$I(q)$')

    plt.tight_layout()
    plt.savefig(open(wdir+'snapshot_%d.png'%window,'wb'))
    # exit()

    def animate(i):
       # cax.set_array(I[i])
       cax.set_array(I[i])
       ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
       cax2.set_array(IQ[i])
       # ax2.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
       ax2.set_title(r'$\theta = %.2f$'%(thetaQ[i]))
       cax3.set_array(IQa[i])
       # ax3.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
       ax3.set_title(r'$\alpha = %.2f$'%(-phase_t[i]/2))

    # if n_win > 100:
    #     nframes = 100
    #     frames = np.linspace(0,len(IQa),num=nframes,endpoint=False,dtype=int)
    #     print(frames)
    #     anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=frames)
    # else:
    anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=n_win)
    # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=np.min((n_win,20)))
    # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=n_win)
    anim.save(wdir+'IqRQa_%d_%d.mp4'%(ntraj,nbins))

    # # Gdot = gfunc(teval)
    # print(Gdot.shape)
    # Gdot_corot = np.zeros(Gdot.shape)
    # Gdot_phase = np.zeros(Gdot.shape)

    # for i in range(0,len(theta)):
    #     # if abs(theta[i] - theta[i-1]) > np.pi:
    #     # print(i,tflow[i],theta[i],theta[i-1],np.round((theta[i] - theta[i-1])/(2*np.pi)))
    #     # theta[i] += 2*np.pi
    #     # theta[i] -= 2*np.pi*np.round((theta[i] - theta[i-1])/(2*np.pi))
    #     # phase = -phase_t_lab[i]*90.0/np.pi
    #     # phase = -phase_t_lab[i]/2.0
    #     phase = -phase_t[i]/2.0 + thetaQ[i]
    #     phaseQ = thetaQ[i]
    #     R = np.array([[np.cos(phase),-np.sin(phase),0.0],[np.sin(phase),np.cos(phase),0.0],[0.0,0.0,0.0]])
    #     RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,0.0]])
    #     # print(i,phase_t[i])
    #     # print(R)
    #     Gdot_corot[i] = np.transpose(RQ)@Gdot[i]@RQ
    #     # Gdot_corot[i] = Q[i]@Gdot[i]@np.transpose(Q[i])
    #     Gdot_phase[i] = np.transpose(R)@Gdot[i]@R

    # GdotT = np.transpose(Gdot,axes=[0,2,1])
    # GdotT_phase = np.transpose(Gdot_phase,axes=[0,2,1])
    # GdotT_corot = np.transpose(Gdot_corot,axes=[0,2,1])

    # E = 0.5*(Gdot + GdotT)
    # Ep = 0.5*(Gdot_phase + GdotT_phase)
    # Ec = 0.5*(Gdot_corot + GdotT_corot)

    # # Visualize on grid
    # w = 3
    # ngrid = 25
    # grid = np.linspace(-w,w,ngrid)
    # Y, X = np.mgrid[-w:w:np.complex(ngrid), -w:w:np.complex(ngrid)]
    # # Y, X = np.meshgrid(grid, grid)
    # U = np.zeros((len(Gdot),ngrid,ngrid))
    # V = np.zeros((len(Gdot),ngrid,ngrid))
    # UE = np.zeros((len(Gdot),ngrid,ngrid))
    # VE = np.zeros((len(Gdot),ngrid,ngrid))
    # UEp = np.zeros((len(Gdot),ngrid,ngrid))
    # VEp = np.zeros((len(Gdot),ngrid,ngrid))
    # UEc = np.zeros((len(Gdot_corot),ngrid,ngrid))
    # VEc = np.zeros((len(Gdot_corot),ngrid,ngrid))
    # for i in range(len(U)):
    #     U[i] = Gdot[i,0,0]*X + Gdot[i,0,1]*Y
    #     V[i] = Gdot[i,1,0]*X + Gdot[i,1,1]*Y
    #     UE[i] = E[i,0,0]*X + E[i,0,1]*Y
    #     VE[i] = E[i,1,0]*X + E[i,1,1]*Y
    #     UEp[i] = Ep[i,0,0]*X + Ep[i,0,1]*Y
    #     VEp[i] = Ep[i,1,0]*X + Ep[i,1,1]*Y
    #     UEc[i] = Ec[i,0,0]*X + Ec[i,0,1]*Y
    #     VEc[i] = Ec[i,1,0]*X + Ec[i,1,1]*Y

    # speed = np.sqrt(U**2 + V**2)
    # Enorm = np.sqrt(UE**2 + VE**2)
    # Epnorm = np.sqrt(UEp**2 + VEp**2)
    # Ecnorm = np.sqrt(UEc**2 + VEc**2)
    # print(np.min(Enorm),np.max(Enorm))
    # print(speed.shape)
    # # exit()

    # # UE = E[0,0]*X + E[0,1]*Y
    # # VE = E[1,0]*X + E[1,1]*Y
    # # Enorm = np.sqrt(UE**2 + VE**2)
    # # # vE = E@

    # # UW = W[0,0]*X + W[0,1]*Y
    # # VW = W[1,0]*X + W[1,1]*Y
    # # Wnorm = np.sqrt(UW**2 + VW**2)

    # # cmin = np.min([speed,Enorm,Wnorm])
    # cmin = 0
    # # cmax = np.max([speed,Enorm,Wnorm])
    # cmax = np.max([speed])
    # print(cmin,cmax)

    # fig = plt.figure(figsize=(10.5,3),dpi=200)

    # ax1 = fig.add_subplot(131)
    # cax1 = ax1.pcolormesh(X,Y,Enorm[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
    # cb = fig.colorbar(cax1,ax=ax1)
    # # cb.ax.set_title(r'$|\nabla \boldsymbol{v}|$')
    # cb.ax.set_title(r'$|\textbf{E}|$')
    # ax1.set_xlabel(r'$x$')
    # ax1.set_ylabel(r'$y$')
    # ax1.set_xlim(-3,3)
    # ax1.set_ylim(-3,3)
    # ax1.set_aspect('equal')

    # ax2 = fig.add_subplot(132)
    # cax2 = ax2.pcolormesh(X,Y,Ecnorm[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
    # cb = fig.colorbar(cax2,ax=ax2)
    # cb.ax.set_title(r'$|\textbf{E}_{c}|$')
    # ax2.set_xlabel(r'$x$')
    # ax2.set_xlim(-3,3)
    # ax2.set_ylim(-3,3)
    # ax2.set_aspect('equal')

    # ax3 = fig.add_subplot(133)
    # cax3 = ax3.pcolormesh(X,Y,Epnorm[0],cmap='inferno',shading='gouraud',vmin=cmin,vmax=cmax)
    # cb3 = fig.colorbar(cax3,ax=ax3)
    # cb3.ax.set_title(r'$|\textbf{E}_{\alpha}|$')
    # ax3.set_xlabel(r'$x$')
    # ax3.set_xlim(-3,3)
    # ax3.set_ylim(-3,3)
    # ax3.set_aspect('equal')

    # ax1.streamplot(X,Y,UE[0],VE[0],color='white',linewidth=1.0)
    # ax2.streamplot(X,Y,UEc[0],VEc[0],color='white',linewidth=1.0)
    # ax3.streamplot(X,Y,UEp[0],VEp[0],color='white',linewidth=1.0)

    # plt.tight_layout()
    # # plt.show()
    # # exit()

    # def animate(i):

    #     # ax3.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))
    #     # ax1.set_title(r'$\alpha = %.2f$'%(-phase_t_lab[i]*90.0/np.pi))
    #     # ax3.set_title(r'$|\textbf{E}_1 - \textbf{E}_2| = %.2e$'%np.mean(Enorm - Ecnorm)**2)
    #     # print(ax.collections)
    #     # ax.collections = [] # clear lines streamplot
    #     # print(ax.collections[0],ax.collections[1])
    #     # exit()
    #     ax1.collections = [ax1.collections[0]]
    #     ax2.collections = [ax2.collections[0]]
    #     ax3.collections = [ax3.collections[0]]
    #     # ax.collections = [ax.collections[0],ax.collections[1],ax.collections[2]]
    #     # print(ax.collections)
    #     # exit()

    #     # Clear arrowheads streamplot.
    #     for artist in ax1.get_children():
    #         if isinstance(artist, FancyArrowPatch):
    #             artist.remove()
    #     for artist in ax2.get_children():
    #         if isinstance(artist, FancyArrowPatch):
    #             artist.remove()
    #     for artist in ax3.get_children():
    #         if isinstance(artist, FancyArrowPatch):
    #             artist.remove()

    #     # stream = ax.streamplot(X,Y,U[i],V[i],color='black')
    #     ax1.streamplot(X,Y,UE[i],VE[i],color='white',linewidth=1)
    #     ax2.streamplot(X,Y,UEc[i],VEc[i],color='white',linewidth=1)
    #     ax3.streamplot(X,Y,UEp[i],VEp[i],color='white',linewidth=1)
    #     cax1.set_array(Enorm[i])
    #     cax2.set_array(Ecnorm[i])
    #     cax3.set_array(Epnorm[i])
    #     print(i)
    #     # return stream

    # nframes = 100
    # frames = np.linspace(0,len(IQa),num=nframes,endpoint=False,dtype=int)
    # print(frames)
    # anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=frames)
    # # anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(Gdot), interval=100, blit=False, repeat=False)
    # # anim.save('test_rotEeig.mp4')
    # # anim.save('test_rotm1.mp4')
    # # anim.save('test_rotGdoteig.mp4')
    # anim.save(wdir+'flow_rotphase_%d.mp4'%len(Gdot))
    # # anim.save(wdir+'flow_norot.mp4')





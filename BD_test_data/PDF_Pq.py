#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import time

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

import scipy
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.io import savemat
from scipy import special

def cyl_pq(qx,qy,qz,R,L,gamma,beta,alpha):

    return (4.0*scipy.special.j1(R*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5))*np.sin(0.5*L*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta)))/(L*R*(qz*np.cos(beta) + (qx*np.cos(alpha) + qy*np.sin(alpha))*np.sin(beta))*((qy*np.cos(alpha) - qx*np.sin(alpha))**2 + (np.cos(beta)*(qx*np.cos(alpha) + qy*np.sin(alpha)) - qz*np.sin(beta))**2)**(0.5)))**2


###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    mymap=matplotlib.cm.get_cmap('jet')

    # FRR = float(sys.argv[1])
    # Q2 = float(sys.argv[2]) # Divided by 1e-7 m3/s
    # strm = int(sys.argv[3])
    flowType = float(sys.argv[1])
    Pe = float(sys.argv[2])
    # flow_step = float(sys.argv[3])
    # strain_max = float(sys.argv[3])
    avg_win = float(sys.argv[3])
    nqi = int(sys.argv[4])
    qlim = float(sys.argv[5])
    nbins = int(sys.argv[6])
    ntraj = int(sys.argv[7])
    load = int(sys.argv[8])

    data = np.loadtxt('Gamma_input/FT%.2f_Pe%.1e.txt'%(flowType,Pe),skiprows=2)
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

    print('No. time steps ',tmax)

    # ntraj = 1000
    dt = 0.0005

    # L = 9200;
    # R = 33;
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
            # Qx[Qxnum*xcount + ycount] = Qxi
            # Qy[Qxnum*xcount + ycount] = Qyi
            ycount = ycount + 1
        xcount = xcount + 1

    # print(Qx[0:10])
    # print(Qy[0:10])
    gamma = 0.0

    # ntraj = 100
    samp_rate = 1
    # avg_win = 0.05 # BD time units between P(q) snapshots
    steps_p_win = int(avg_win/(dt*samp_rate))
    print('Time steps per window %d'%steps_p_win)
    n_win = math.ceil(tend/avg_win)
    # n_win = 2
    print('Number of windows %d'%n_win)
    P = np.zeros((n_win,nQ))
    print('P array size %f GB'%(8.0*P.size/1e9))
    snap_count = np.zeros(n_win)

    # wdir = 'output/FRR_%.2f/Q2_%.2f/strm_%d/'%(FRR,Q2,strm)
    # wdir = 'output/'

    # nbins = 300
    # ang_list = np.linspace(0,np.pi,nbins)
    # bin_area = (ang_list[0] - ang_list[1])**2
    # h = np.zeros((n_win,nbins,nbins))
    # for nw in range(n_win):
        # h[nw] = np.loadtxt(wdir+'ht_%.2f_%.2f_%d_%d_%d.txt'%(FRR,Q2,strm,ntraj,nw))
    wdir = 'output/FT%.2f_Pe%.1e/'%(flowType,Pe)

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
            # dphi = (jp + 0.5 - 1.0*np.pi)*1.0*dangle
            dtheta = (jt + 0.5)*dangle
            for jp in range(nbins):
                dphi = (jp + 0.5)*dangle
                Pj = cyl_pq(Qx,Qy,Qz,R,L,gamma,dtheta,dphi)
                # print('theta %f phi %f'%(dtheta,dphi))
                # for k in range(len(Pj)):
                #     print('%d %lf'%(k,Pj[k]))
                # exit()
                for nw in range(n_win):
                    cum_weight[nw] += 2.0*dangle*dangle*np.sin(dtheta)*h[nw][jt][jp]
                    # print(Pj.shape)
                    # print(h[nw][jt][jp].shape)
                    # print(P[nw].shape)
                    P[nw] += 2.0*dangle*dangle*np.sin(dtheta)*h[nw][jt][jp]*Pj
                    # P[nw] += 2.0*dangle*dangle*np.sin(dtheta)*h[jt][jp]*Pj

            print("--- jt %d dtheta %f %s seconds ---" % (jt,dtheta,time.time() - start_time))

        print("--- %s seconds ---" % (time.time() - start_time))
        I = 1.0e8*np.pi*R**2*L*volfrac*dsld**2*P/cum_weight[:,np.newaxis] + bkgd
        P = np.reshape(P,(n_win,Qxnum,Qynum))
        I = np.reshape(I,(n_win,Qxnum,Qynum))
            
        pickle.dump([I,cum_weight],open(wdir+'Iq_%d_%d.p'%(ntraj,nbins),'wb'))

    else:
        # P,cum_weight = pickle.load(open(wdir+'Pq_%d_%d.p'%(ntraj,nbins),'rb'))
        # I = 1.0e8*np.pi*R**2*L*volfrac*dsld**2*P/cum_weight[:,np.newaxis,np.newaxis] + bkgd
        I,cum_weight = pickle.load(open(wdir+'Iq_%d_%d.p'%(ntraj,nbins),'rb'))

    Iqmin = np.min(I)

    t = np.linspace(0,10,n_win)
    # mymap=matplotlib.cm.get_cmap('inferno')
    mymap=matplotlib.cm.get_cmap('jet')

    fig = plt.figure(figsize=(4,3),dpi=300)
    ax = fig.add_subplot(111)

    # cax = ax.pcolormesh(qxlist,qylist,P[0],norm=colors.LogNorm(vmin=1e-4),cmap=mymap,shading='gouraud')
    cax = ax.pcolormesh(qxlist,qylist,I[0],norm=colors.LogNorm(vmin=Iqmin),cmap=mymap,shading='gouraud')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$q_x [\AA^{-1}]$')
    ax.set_ylabel(r'$q_y [\AA^{-1}]$')
    cb = fig.colorbar(cax,ax=ax)
    cb.ax.set_title(r'$I(q)$')
    plt.tight_layout()

    # fig.colorbar(cax)

    def animate(i):
       cax.set_array(I[i])
       ax.set_title(r'$\tilde{t} = %.2f$'%(i*avg_win))

    anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)
    anim.save(wdir+'Iq_%d_%d.mp4'%(ntraj,nbins))
    # plt.show()

    exit()
    # # fig,ax = plt.subplots(111)
    # # plt.pcolormesh(qxlist,qylist,P)
    # plt.title(r'$\tilde{t} = %.2f$'%(win*avg_win))
    # # im = ax.pcolormesh(qxlist,qylist,P[win],norm=colors.LogNorm(vmin=1e-4),cmap=mymap,shading='gouraud')
    # im = ax.pcolormesh(qxlist,qylist,P[win],norm=colors.LogNorm(),cmap=mymap,shading='gouraud')
    # ax.set_aspect('equal')
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    # cb = fig.colorbar(im,ax=ax)
    # cb.ax.set_title(r'$P(q)$')

    # # cax = fig.add_axes([.6, 0.03, 0.3, 0.02]) #left bottom width height
    # # cb=fig.colorbar(im, cax=cax, orientation='vertical')
    # # cb.set_label(r'$u$',horizontalalignment='right')
    # # cb.ax.set_title(r'$P(q)$',loc='right')

    # plt.savefig(open(outdir+'/window%d.png'%(win),'wb'))
    # # plt.show()
    # plt.close()



        


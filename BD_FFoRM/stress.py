#! /home/cyoung37/anaconda3/bin/python3

# Computes stress along Lagrangian streamlines using uu, uuuu from BD simulations

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

def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    # if you want it in a list of indices . . . 
    # return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose().tolist()

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    mymap=matplotlib.cm.get_cmap('jet')

    FRR = float(sys.argv[1])
    Q2 = float(sys.argv[2]) # Divided by 1e-7 m3/s
    strm = int(sys.argv[3])
    # avg_win = float(sys.argv[4])
    # # nqi = int(sys.argv[5])
    # # qlim = float(sys.argv[6])
    # # nbins = int(sys.argv[7])
    # ntraj = int(sys.argv[5])

    avg_win = float(sys.argv[4])
    ntraj = int(sys.argv[5])

    L = 1680.0
    R = 34.0
    Dr = 2.0
    AR = L/(2.0*R)*np.sqrt(8.0*np.pi/(16.35*np.log(L/(2.0*R))))

    A = AR**2/(2.0*(np.log(2.0*AR) - 1.5))
    B = (6.0*np.log(2.0*AR) - 11.0)/AR**2
    C = 2
    F = 3*AR**2/(np.log(2.0*AR) - 0.5)
    # print(A,B,C,F)

    # step_size = 0.05
    # FRR = np.arange(-0.7,1.00001,step_size)
    # FRR = FRR[::2]
    # # Q2 = np.array([3.4e-7])
    # # Q2 = Q2/1.0e-7
    # # strm = np.array([4])
    # Q2max = 1.0e-6
    # Q2min = 1.0e-8
    # Q2arr = np.linspace(Q2min,Q2max,10)
    # Q2 = Q2arr[1::2]
    # Q2 = Q2/1.0e-7
    # strm = np.arange(0,10)

    FRR = np.array([FRR])
    Q2 = np.array([Q2])
    strm = np.array([strm])

    for f in FRR:
        for q in Q2:
            for st in strm:
                data = np.loadtxt('../Gamma_input_upstream/FRR_%.2f/Q2_%.2f/strm_%d.txt'%(f,q,st),skiprows=2)
                tflow = data[:,0]
                flow_type = data[:,1]
                flow_strength = data[:,2]
                Gdot = data[:,3:]
                Gdot = np.reshape(Gdot,(Gdot.shape[0],3,3))
                tflow = tflow*Dr
                Gdot = Gdot/Dr
                dt = 0.0005
                tend = tflow[-1]
                tmax = int(tend/dt)

                print('No. time steps ',tmax)

                # ntraj = 100
                samp_rate = 1
                # avg_win = 0.05 # BD time units between P(q) snapshots
                steps_p_win = int(avg_win/(dt*samp_rate))
                print('Time steps per window %d'%steps_p_win)
                n_win = math.ceil(tend/avg_win)
                # n_win = 2
                print('Number of windows %d'%n_win)

                wdir = 'output_upstream/FRR_%.2f/Q2_%.2f/strm_%d/uu_u4/'%(f,q,st)

                # h = np.loadtxt(wdir+'h_na_%d_nq_%d_qlim_%.2f.txt'%(nbins,Qxnum,Qxmax))
                data = np.loadtxt(wdir+'uu_%d.txt'%(ntraj),skiprows=1)
                t = data[:,0]
                uu = data[:,1:]
                data = np.loadtxt(wdir+'u4_%d.txt'%(ntraj),skiprows=1)
                t = data[:,0]
                u4 = data[:,1:]

                # print(uu.shape)
                # print(u4.shape)

                # Block averaged orientation moment tensors
                uu_ba = np.zeros((n_win,9))
                u4_ba = np.zeros((n_win,81))
                Gd_ba = np.zeros((n_win,3,3))

                for i in range(n_win):
                    uu_ba[i] = np.mean(uu[i*steps_p_win:(i+1)*steps_p_win],axis=0)
                    u4_ba[i] = np.mean(u4[i*steps_p_win:(i+1)*steps_p_win],axis=0)

                tsnap = np.arange(0,n_win)*avg_win
                for i in range(n_win):
                    tdiff = np.abs(tflow - tsnap[i])
                    ind = get_indices_of_k_smallest(tdiff,2)
                    # print(i,tsnap[i],ind[0][0],ind[0][1],tflow[ind],Gdot[ind[0][0]],Gdot[ind[0][1]])
                    Gd_ba[i] = (Gdot[ind[0][0]] - Gdot[ind[0][1]])/(tflow[ind[0][0]] - tflow[ind[0][1]])*(tsnap[i] - tflow[ind[0][0]]) + Gdot[ind[0][0]]

                # exit()

                # count = 0
                # win = 0

                # for i in range(len(Gdot)):
                #     Gd_ba[win] += Gdot[i]
                #     count += 1
                #     if tflow[i] > (win+1)*avg_win:
                #         Gd_ba[win] /= count
                #         print('win %d count %d tflow %f Gd_ba_00 %f'%(win,count,tflow[i],Gd_ba[win,0,0]))
                #         count = 0
                #         win += 1

                # exit()

                Gd_tr = np.transpose(Gd_ba,axes=[0,2,1])
                E_ba = 0.5*(Gd_ba + Gd_tr)

                uu_ba = np.reshape(uu_ba,(uu_ba.shape[0],3,3))
                u4_ba = np.reshape(u4_ba,(u4_ba.shape[0],3,3,3,3))
                eye = np.zeros((n_win,3,3))
                eye[:] = np.identity(3)

                IS = np.einsum('ijk,ilm->ijklm',eye,uu_ba)
                ISddE = np.einsum('ijklm,ilm->ijk',IS,E_ba)

                term1 = A*(np.einsum('ijklm,ilm->ijk',u4_ba,E_ba))
                # term1 = A*(- 1/3*ISddE)
                term2 = Dr*F*(uu_ba - 1/3*eye)

                stress = 2.0*(term1 + term2)
                # stress = 2.0*(term2)
                tout = np.arange(0,n_win)*avg_win

                train_out_dir = 'training_output/stress_ups_L%.1f_R%.1f/'%(L,R)
                if os.path.exists(train_out_dir) == 0:
                    os.makedirs(train_out_dir,exist_ok=False)

                # u4_ba = np.reshape(u4_ba,(u4_ba.shape[0],81))
                # for i in range(81):
                #     plt.figure(figsize=(4,3),dpi=200)
                #     plt.plot(tout,u4_ba[:,i])
                #     plt.xlabel(r'$t$')
                #     plt.ylabel(r'$S^{(4)}_{%d}$'%i)
                #     plt.tight_layout()

                # plt.show()
                # exit()

                # plt.figure(figsize=(4,3),dpi=200)
                # plt.plot(tout,stress[:,0,0])
                # plt.xlabel(r'$t$')
                # plt.ylabel(r'$\sigma_{p,xx}$')
                # plt.tight_layout()
                # plt.show()
                # exit()

                # fig = plt.figure(figsize=(8,4),dpi=300)
                # ax = plt.subplot(231)
                # ax.plot(tout,E_ba[:,0,0])
                # ax.set_ylabel(r'$S_{xx}$')
                # # ax.set_xlabel(r'$t$')

                # ax = plt.subplot(232)
                # ax.plot(tout,E_ba[:,0,1])
                # ax.set_ylabel(r'$S_{xy}$')
                # # ax.set_xlabel(r'$t$')

                # ax = plt.subplot(233)
                # ax.plot(tout,E_ba[:,0,2])
                # ax.set_ylabel(r'$S_{xz}$')
                # # ax.set_xlabel(r'$t$')

                # ax = plt.subplot(234)
                # ax.plot(tout,E_ba[:,1,1])
                # ax.set_ylabel(r'$S_{yy}$')
                # # ax.set_xlabel(r'$t$')

                # ax = plt.subplot(235)
                # ax.plot(tout,E_ba[:,1,2])
                # ax.set_ylabel(r'$S_{yz}$')
                # # ax.set_xlabel(r'$t$')

                # ax = plt.subplot(236)
                # ax.plot(tout,E_ba[:,2,2])
                # ax.set_ylabel(r'$S_{zz}$')
                # # ax.set_xlabel(r'$t$')

                # plt.tight_layout()
                # # plt.savefig(open(model_path+'predictions/FRR_%.2f_Q2_%.2f_strm_%d_s%d.png'%(f,q,st,comp),'wb'))
                # plt.show()
                # # plt.close()

                # for i in range(10):
                #     print(i,stress[i,0,1],E_ba[i,0,0])

                # exit()
                pickle.dump([tout,stress,Gd_ba,E_ba],open(train_out_dir+'FRR_%.2f_Q2_%.2f_strm_%d.p'%(f,q,st),'wb'))

    # print(stress.shape)
    # for i in range(n_win):
        # print(i,stress[i])


    # exit()

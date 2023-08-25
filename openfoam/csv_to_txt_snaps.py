#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

# This code collects Lagrangian streamlines near the stagnation point and generates the velocity gradient time series to be input to the BD simulations for creating synthetic scattering data

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    # FRR = float(sys.argv[1])
    # Q2 = float(sys.argv[2]) # Divided by 1e-7 m3/s
    R2 = 0.0025**2
    n_strm = 10

    dct = np.loadtxt('snapshots/dictionary.txt',skiprows=2)
    snap_id = dct[:,0] - 1
    # print(snap_id)
    # exit()
    # snap_id = dct[:,0]
    # snap_id = dct[:,0] - 351
    FRR_id = dct[:,1]
    Q2_id = dct[:,2]
    # snapshots = np.arange(0,350)
    # snapshots = np.arange(0,349)
    # snapshots = np.arange(0,35)
    # snapshots = [349]
    # for snap in snapshots:
    for snap in snap_id[1:]:
        # print(snap)
        snap = int(snap)
        # data = np.genfromtxt('FRR_%.2f/strm_20.csv'%FRR, delimiter=',', names=True)
        data = np.genfromtxt('snapshots/strm_200_%d.csv'%snap, delimiter=',', names=True)
        # print(data)
        out_path = '../../BD_FFoRM/Gamma_input_test/FRR_%.2f/Q2_%.2f/'%(FRR_id[snap + 1],Q2_id[snap + 1])
        if os.path.exists(out_path) == 0:
            os.makedirs(out_path)
        out_path_plots = 'snapshots/plots3/'
        if os.path.exists(out_path_plots) == 0:
            os.makedirs(out_path_plots)

        streamline_ids = np.unique(data['SeedIds']).astype(int)

        tend = np.zeros(len(streamline_ids))
        min_r = np.zeros(len(streamline_ids))
        for i in streamline_ids[:-1]:
            # print(i)
            rows = np.where(data['SeedIds'] == i)
            x = data['Points0'][rows]
            y = data['Points1'][rows]
            data2 = data[:][rows]
            indp = np.where(x**2 + y**2 < R2)[0]
            # print(indp)
            if not indp.size:
                continue
            x = x[indp]
            y = y[indp]
            r2 = x**2 + y**2
            mr = np.min(r2)
            # print(min_r)
            min_r[i] = mr
            data2 = data2[:][indp]
            t = data2['IntegrationTime']
            t = t - t[0]
            tend[i] = t[-1]

        min_r = min_r[min_r!=0]
        ind_min = np.argsort(min_r)
        # print(ind_min)
        # print(min_r[ind_min])
        # print(ind_min[0:20])
        n_sort_indmin = np.sort(ind_min[0:n_strm])
        # exit()
        # print(min_r)
        # exit()
        # min_r_ind = np.argmin(min_r)
        # print(min_r_ind,min_r[min_r_ind])
        # ind_lt = np.argpartition(tend,-20)[-20:]
        # exit()
        # print(min_r_ind)
        # exit()
        # ind_lt = np.argpartition(tend,-20)[-20:]
        # print(ind_lt)
        # ind_sort = ind_lt[np.argsort(tend[ind_lt])]
        # print(ind_sort)
        # print(tend[ind_sort])
        # exit()
        # print(np.sort(ind_lt))
        # exit()
        # ind_lt = np.sort(ind_lt)
        # ind_out = np.arange(ind_sort[-1] - 10,ind_sort[-1] + 10)
        # ind_out = np.arange(min_r_ind - 10,min_r_ind + 10)
        ind_out = n_sort_indmin

        count = 0
        # for i in streamline_ids:
        for i in ind_out:
            print(i)
            rows = np.where(data['SeedIds'] == i)
            x = data['Points0'][rows]
            y = data['Points1'][rows]
            data2 = data[:][rows]
            indp = np.where(x**2 + y**2 < R2)[0]
            # print(indp)
            if not indp.size:
                continue
            x = x[indp]
            y = y[indp]
            data2 = data2[:][indp]
            t = data2['IntegrationTime']
            t = t - t[0]
            Gd = np.zeros((len(t),9))
            for j in range(9):
                Gd[:,j] = data2['gradU%d'%j]
            # print(Gd[:,0])
            Gd[:,2] = 0.0
            Gd[:,5] = 0.0
            Gd[:,6:8] = 0.0
            Gd = np.reshape(Gd,(len(Gd),3,3))

            Gd_tr = np.transpose(Gd,axes=[0,2,1])
            vort = 0.5*(Gd - Gd_tr)
            E = 0.5*(Gd + Gd_tr)
            vmag = np.einsum('ijk,ijk->i',vort,vort)**0.5
            Emag = np.einsum('ijk,ijk->i',E,E)**0.5
            flow_type = (Emag - vmag)/(Emag + vmag)
            flow_type[flow_type == np.inf] = 0
            GamMag = np.einsum('ijk,ijk->i',Gd,Gd)**0.5
            G = GamMag/np.sqrt(1 + flow_type**2)

            plt.figure(figsize=(12,3))
            ax = plt.subplot(131)
            plt.scatter(x,y)
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')

            ax = plt.subplot(132)
            plt.plot(t,flow_type)
            plt.xlabel('Time (s)')
            plt.ylabel('Flow type parameter '+r'$\Lambda$')

            ax = plt.subplot(133)
            plt.plot(t,G)
            plt.xlabel('Time (s)')
            plt.ylabel('Flow strength G (1/s)')

            plt.tight_layout()
            plt.savefig(open(out_path_plots+'FTP_G_%.2f_%.2f_%d_%d.png'%(FRR_id[snap + 1],Q2_id[snap + 1],snap + 1,i),'wb'))
            plt.close()

            fname = out_path+'strm_%d.txt'%count
            newfile=open(fname,'w+')
            newfile.write('Time ')
            newfile.write('ftype ')
            newfile.write('fmag ')
            for i in range(3):
                for j in range(3):
                    newfile.write('G_%d%d '%(i,j))
            newfile.write('\n')
            newfile.write('---------------------\n')
            # newfile.close()

            Gd = np.reshape(Gd,(len(Gd),9))
            for j in range(len(t)):
                # print(i,time_elap,f_step)
                newfile.write('%f '%t[j])
                newfile.write('%f '%flow_type[j])
                newfile.write('%f '%G[j])
                for k in range(9):
                    # for k in range(3):
                    newfile.write('%f '%Gd[j,k])
                newfile.write('\n')

            newfile.close()

            count = count + 1

            # if i==streamline_ids[-1]:
                # break # last streamline is labeled nan, some paraview bug

    # plt.show()

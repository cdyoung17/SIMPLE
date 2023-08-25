#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import time

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    flowType = float(sys.argv[1])
    Pelist = np.array([float(sys.argv[2])])
    strain_max = float(sys.argv[3])
    # Pelist = np.logspace(0,np.log10(50.0),10)
    # low = np.array([0.1,0.5])
    # Pelist = np.append(low,Pelist)
    print(Pelist)
    # flow_step = float(sys.argv[3])
    # tmax = float(sys.argv[3])

    for Pe in Pelist:
        flow_step = 5.0e-4 # BD simulation time step
        tmax = int(np.ceil(strain_max/Pe))
        steps = int(np.ceil(tmax/flow_step))
        print('Pe %.2e Time units %.2f steps %d'%(Pe,tmax,steps))

        t = np.arange(0,steps)*flow_step
        # ftype = np.ones(steps)*flowType
        # Gd = np.zeros((steps,9))
        # Gd[:,0] = Pe*(1 + flowType)
        # Gd[:,1] = Pe*(1 - flowType)
        # Gd[:,3] = -Pe*(1 - flowType)
        # Gd[:,4] = -Pe*(1 + flowType)
        Gd = np.array([Pe*(1 + flowType), Pe*(1 - flowType),0.0,
                        -Pe*(1 - flowType),-Pe*(1 + flowType),0.0,
                        0.0,0.0,0.0])
        # Gd = np.array([-5.0,-30.0,0.0,
                        # -30.0,5.0,0.0,
                        # 0.0,0.0,0.0])

        fname = 'Gamma_input/FT%.2f_Pe%.1e.txt'%(flowType,Pe)
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

        # Gd = np.reshape(Gd,(len(Gd),9))
        for j in range(len(t)):
            # print(i,time_elap,f_step)
            newfile.write('%f '%t[j])
            newfile.write('%f '%flowType)
            newfile.write('%f '%Pe)
            for k in range(9):
                # for k in range(3):
                # newfile.write('%f '%Gd[j,k])
                newfile.write('%f '%Gd[k])
            newfile.write('\n')

        newfile.close()
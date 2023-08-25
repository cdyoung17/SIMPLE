#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import numpy as np
from distutils.dir_util import copy_tree

# This code extracts the last ts from the simulations (presumably steady state) and numbers them so they can be loaded into paraview sequentially for applying the streamline generation pipeline

if __name__=='__main__':

	dir1 = 'template'
	step_size = 0.05
	# Sweep parameter - flow rate ratio controls flow type, ext vs shear
	FRR = np.arange(-0.7,1.00001,step_size)
	# FRR = [-0.05,-0.10]
	# ind_remove = [13,14,15]
	# FRR = np.delete(FRR,ind_remove)
	# print(FRR)
	# exit()
	nF = len(FRR)
	nQ = 10
	# nQ = 1
	# Boundary conditions - inlet and outlet flow rates
	Q1 = np.zeros((nF,nQ))
	Q2 = np.zeros((nF,nQ))
	Q2max = 1.0e-6
	Q2min = 1.0e-8
	Q2[:] = np.linspace(Q2min,Q2max,nQ)
	# Q2[:] = [1.0e-6]
	# print(Q2)
	# exit()
	# Increase total flow rate at lower values of FRR to maintain similar strain rate at stagnation pt
	# 5x increase in Q2 from FRR = 1.0 to FRR = -0.7 used in Corona et al. Phys. Rev. Mat. 2022
	# Q2[-1] = 8.33333e-7
	# Q2[0] = Q2[-1]*5
	# Linear interpolation to find other total flow rates
	# Q2[1:-1] = Q2[0] + (FRR[1:-1] - FRR[0])*(Q2[-1] - Q2[0])/(FRR[-1] - FRR[0])

	for i in range(nF):
		Q1[i] = -Q2[i]*FRR[i]

	outdir = 'snapshots/'
	if(os.path.exists(outdir)==0):
		os.makedirs(outdir)

	dictstr = outdir+'dictionary.txt'
	dictfile = open(dictstr,'w+')
	dictfile.write('Snapshot FRR Q2\n')
	dictfile.write('-----------------\n')
	# print(Q1)
	# Create directories. Copy contents and rewrite 0/U for new BC
	# count = 351
	count = 0
	for i in range(nF):
		for j in range(nQ):
			# print(i,FRR[i],Q1[i],Q2[i])
			dir1 = 'FRR_%.2f/Q2_%.2f/'%(FRR[i],Q2[i,j]/1.0e-7)
			print(dir1)
			# os.chdir(dir1)
			dir_list = [name for name in os.listdir(dir1) if os.path.isdir(dir1+name)]
			# print(dir_list)
			# print(float(dir_list[0]))
			# print(np.longdouble(dir_list[0]))
			# print(dir_list)
			snaps = []
			for k in dir_list:
				if k[0].isdigit():
					# snaps.append(float(k))
					snaps.append(np.longdouble(k))
			print(snaps)
			last_snap = snaps[np.argmax(snaps)]
			# print('i %d FRR %.5f Q1 %.4e Q2 %.4e dir2 %s'%(i,FRR[i],Q1[i,j]/1.0e-7,Q2[i,j]/1.0e-7,dir2))
			# if(os.path.exists(dir2)==0):
				# os.makedirs(dir2)
			# last_snap_dir = dir_list[1]
			last_snap_dir = dir1+str(last_snap)
			# print(last_snap_dir)
			# copy_dir = dir1+last_snap_dir
			# print(copy_dir,count)
			# os.chdir('../..')
			copy_tree(last_snap_dir,outdir+'%d'%count)
			dictfile.write('%d %.2f %.2f\n'%(count,FRR[i],Q2[i,j]/1.0e-7))
			count = count + 1
			# os.chdir('../..')
			# exit()

	dictfile.close()
#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import numpy as np
from distutils.dir_util import copy_tree

if __name__=='__main__':

	dir1 = 'template'
	step_size = 0.05
	# Sweep parameter - flow rate ratio controls flow type, ext vs shear
	FRR = np.arange(-0.7,1.00001,step_size)
	nQ = len(FRR)
	# Boundary conditions - inlet and outlet flow rates
	Q1 = np.zeros(nQ)
	Q2 = np.zeros(nQ)
	# Increase total flow rate at lower values of FRR to maintain similar strain rate at stagnation pt
	# 5x increase in Q2 from FRR = 1.0 to FRR = -0.7 used in Corona et al. Phys. Rev. Mat. 2022
	Q2[-1] = 8.33333e-7
	Q2[0] = Q2[-1]
	# Linear interpolation to find other total flow rates
	Q2[1:-1] = Q2[0] + (FRR[1:-1] - FRR[0])*(Q2[-1] - Q2[0])/(FRR[-1] - FRR[0])
	Q1 = -Q2*FRR
	# Create directories. Copy contents and rewrite 0/U for new BC
	for i in range(nQ):
		# print(i,FRR[i],Q1[i],Q2[i])
		dir2 = 'FRR_sweep2/FRR_%.2f'%FRR[i]
		os.system('touch %s/open.foam'%dir2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:32:43 2022

@author: kevin
"""
import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle as p
import re
import seaborn as sns
import scipy.io

import torch
import torch.nn as nn
import torch.optim as optim
import torch as T
# import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser('movie')
parser.add_argument('--dPCA', type=int, default=50) # Number of PCA modes input dim
# parser.add_argument('--arch', type=int, default=0) # Architecture label
parser.add_argument('--model', type=int, default=0) # Model #
parser.add_argument('--M', type=int, default=106663) # Number of snapshots
parser.add_argument('--nlin', type=int, default=4) # Number of linear layers in latent space
parser.add_argument('--cd', type=int, default=20) # Linear layer width
parser.add_argument('--wtdc', type=float, default=5e-3) # Weight decay strength
parser.add_argument('--optim', type=str, default='AdamW') # Optimizer string, Adam or AdamW
parser.add_argument('--downs', type=int, default=10) # Epochs per print SV spectrum
parser.add_argument('--train', type=int, default=1) # Train or load pretrained
parser.add_argument('--dh', type=int, default=10) # Latent dimension for output
# parser.add_argument('--dir', type=str, default='') # Directory to find input data, preprocessing methods vary
args = parser.parse_args()

class autoencoder(nn.Module):
    def __init__(self, ambient_dim=30, code_dim=20, nlin=4, filepath='testae'):
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
            # print(l[1])
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

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.write('\n')
    newfile.close()

# Get Covariance, SVD
def getSVD(code_data):
    #Compute covariance matrix and singular values
    code_mean = code_data.mean(axis=0)
    code_std = code_data.std(axis=0)
    #code_data = (code_data - code_mean)/code_std
    code_data = (code_data - code_mean)
    
    #covMatrix = np.cov(code_data.T,bias=False)
    covMatrix = (code_data.T @ code_data) / len(dataset)
    u, s, v = np.linalg.svd(covMatrix, full_matrices=True)

    return code_mean, code_std, covMatrix, u, s, v

if __name__ == '__main__':

    #Parameters
    num_epochs = 2000
    batch_size = 128
    learning_rate = 1e-3
    it_per_drop = int(num_epochs/2)
    dPCA = args.dPCA
    modl = args.model
    M = args.M
    train = args.train
    wt_dc = args.wtdc
    nlin = args.nlin
    code_dim = args.cd
    dh = args.dh

    print('weight decay',wt_dc)
    #Initialize Model
    model = autoencoder(ambient_dim=dPCA,code_dim=code_dim,nlin=nlin).to(device)
    model.double()
    loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000001)
    # wt_dc = 5.0e-9
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_dc)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wt_dc)
    else:
        print('Optimizer untested, create new elif statement and give it a try')
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=it_per_drop, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=it_per_drop, gamma=0.1)
    #load data
    data_dir = '../../BD_FFoRM/preprocessed/na200_nq_100_qlim_0.10_L1680.0_R34.0/training_data/'
    print(data_dir)
    rawdata = p.load(open(data_dir+'red_aligned_snaps_clearnan_%d_%d.p'%(dPCA,M),'rb'))

    wdir = 'dPCA%d/%s_w%.2e_l%d/rmodel%d'%(dPCA,args.optim,wt_dc,nlin,modl)
    if os.path.exists(wdir) == 0:
        os.makedirs(wdir,exist_ok=False)
    os.chdir(wdir)

    if train:
        output('Weight decay %.2e'%wt_dc)
        output('model z value: {:04d}'.format(model.code_dim))
        output('number of linear layers: {:02d}'.format(model.nlin))

    mean = np.mean(rawdata,axis=0)
    std = np.std(rawdata,axis=0)
    std = np.max(std)
    p.dump([mean,std],open('a_stats_v2_%d.p'%dPCA,'wb'))

    #Clean Data
    dataset = (rawdata - mean[np.newaxis,:])/std
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Get Data for Computing SVD
    chunk = np.copy(dataset)
    Testdata = T.tensor(chunk, dtype=T.double).to(device)

    #Initialize Storage Matrices
    tot_err = []
    cov_save = np.array([])
    s_save = np.array([])

    if train:
        for epoch in range(num_epochs):
            for snapshots in dataloader:
                inputs = snapshots
                inputs = inputs.to(device)
                
                #Forward pass, compute loss
                reconstructed = model(inputs)
                loss = loss_function(reconstructed, inputs)
                
                #Back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # exit()
            # ===================log========================
            scheduler.step()
            print('epoch [{}/{}], loss:{:.2e}, lr:{:.6f}'
                .format(epoch + 1, num_epochs, loss.item(), optimizer.param_groups[0]['lr']))
            tot_err.append(loss.item())

            # Print out the Loss and the time the computation took
            if epoch % 10 == 0:
                T.save(model.state_dict(), 'IRMAE_AE_chp.pt')
                output('Iter {:04d} | Total Loss {:.2e} | lr {:.2e}'.format(epoch, loss.item(),optimizer.param_groups[0]['lr']))
                code_data = model.encode(Testdata)
                code_data = code_data.detach().numpy()
                _, _, temp_cov, _, temp_s, _ = getSVD(code_data)
                s_save = np.hstack([s_save, temp_s[:,np.newaxis]]) if s_save.size else temp_s[:,np.newaxis]
                cov_save = np.hstack([cov_save, temp_cov]) if cov_save.size else temp_cov
                p.dump(s_save,open('training_svd.p','wb'))


        T.save(model.state_dict(), 'IRMAE_AE.pt')
        p.dump(tot_err,open('err.p','wb'))
                #Print Training Curve
        fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
        plt.semilogy(tot_err,c='k', label='Total Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('TrainCurve.png')
        
    else:
        if(os.path.exists('IRMAE_AE.pt')) == 1:
            model.load_state_dict(T.load('IRMAE_AE.pt'))
        else:
            model.load_state_dict(T.load('IRMAE_AE_chp.pt'))
        print('testing')
        s_save = p.load(open('training_svd.p','rb'))
        print(s_save.shape)
        
    #Get chunk of data for plotting and computing singular values/basis vectors
    code_data = model.encode(Testdata)
    code_data = code_data.detach().numpy()
    code_mean, code_std, covMatrix, u, s, v = getSVD(code_data)

    reconstructed = model(Testdata)
    loss = loss_function(reconstructed, Testdata).detach().numpy()
    print('Test data MSE %.3e'%loss)
    p.dump(loss,open('test_loss.p','wb'))

    #Save Results
    p.dump([code_mean,code_std],open('code_musigma.p','wb'))
    p.dump([u,s,v],open('code_svd.p','wb'))
    if train:
        p.dump(s_save,open('training_svd.p','wb'))
        p.dump(cov_save,open('training_cov.p','wb'))

    h = code_data @ u[:,:dh]
    print(h.shape)
    p.dump(h,open('encoded_data.p','wb'))

    # fig = plt.figure(figsize=(4,3),dpi=200)
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(code_data[:,0], code_data[:,1], code_data[:,2])
    # ax.set_xlabel(r'$h_0$')
    # ax.set_ylabel(r'$h_1$')
    # ax.set_zlabel(r'$h_2$')
    # plt.tight_layout()
    # plt.savefig(open('3d_proj.png','wb'))
    # plt.show()
    
    #Plotting Results
    #plotting singular values
    fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
    # fig = plt.figure(num=None, figsize=(4,3), dpi=300, facecolor='w', edgecolor='w')
    plt.semilogy(s,'ko--')
    plt.xlabel('singular value rank')
    plt.ylabel('singular value of cov of z')
    plt.tight_layout()
    plt.savefig(open('code_sValues.png','wb'))

    s_save = np.transpose(s_save)
    downs = args.downs
    s_save = s_save[::downs]
    color = iter(cm.inferno(np.linspace(0, 1, len(s_save))))
    fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
    # fig = plt.figure(num=None, figsize=(4,3), dpi=300, facecolor='w', edgecolor='w')
    for i in range(len(s_save)):
        c = next(color)
        plt.semilogy(s_save[i,:],'o--',c=c,label='Epoch %d'%(i*10*downs))
    plt.xlabel('singular value rank')
    plt.ylabel('singular value of cov of z')
    plt.ylim(1e-20,)
    plt.legend()
    plt.tight_layout()
    plt.savefig(open('code_sValues_train.png','wb'))

    #Plotting covariance matrix
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    fig = plt.figure(num=None, figsize=(7, 7), dpi=100, facecolor='w', edgecolor='w')
    plt.imshow(covMatrix,cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(open('cov_matrix.png','wb'))
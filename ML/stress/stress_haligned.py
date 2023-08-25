#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
# sys.path.insert(0, '../')
import gc
import argparse

import numpy as np
import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers

# import scipy.io
# from scipy import signal
# from sklearn.utils.extmath import randomized_svd
# from scipy.integrate import odeint

from sklearn.utils import shuffle

parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--dPCA', type=int, default=50) # Number of PCA modes retained
parser.add_argument('--dh', type=int, default=6) # Autoencoder latent dimension
parser.add_argument('--data_size', type=int, default=106663)  #IC from the simulation
parser.add_argument('--niters', type=int, default=2000)       #Iterations of training
parser.add_argument('--arch', type=int, default=0) # Architecture no.
parser.add_argument('--traj', type=int, default=0) # Model no.
parser.add_argument('--reg', type=float, default=0.0001) # L2 regularization penalty
parser.add_argument('--load', type=int, default=0)
# parser.add_argument('--dir', type=str, default='l2c_smooth') # Name of directory - use for code changes
parser.add_argument('--ae_model', type=int, default=0)
parser.add_argument('--ae_nlin', type=int, default=4)
parser.add_argument('--ae_optim', type=str, default='AdamW')
parser.add_argument('--ae_wtdc', type=float, default=5e-3) # Weight decay strength
args = parser.parse_args()

# Class for printing history
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      newfile=open(name,'a+')
      newfile.write('Epoch: '+str(epoch)+'   ')
      newfile.write('Loss: '+str(logs['loss'])+'\n')
      newfile.close()

###############################################################################
# Plotting Functions
###############################################################################
# For plotting history
def plot_history(histshift):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.semilogy(histshift['epoch'], histshift['loss'],
       label='Train Loss')
    plt.semilogy(histshift['epoch'], histshift['val_loss'],
       label = 'Val Loss')
    plt.legend()
    #plt.ylim([10**-6,10**-1])

def plot_hist(error,title,bins=40,ymax=50,xmax=.05):
    plt.hist(error, bins = bins,density=True)
    plt.xlim([-xmax,xmax])
    plt.ylim([0,ymax])
    plt.xlabel("Prediction Error")
    variance=np.mean(error**2)
    plt.title('MSE='+str(round(variance,7)))
    plt.ylabel(title+' PDF')
    
    return variance

def step_decay(epoch):
    initial_lrate = 0.001
    # initial_lrate = 2.5e-6
    drop = 0.5
    epochs_drop = 100.0
    lr_min = 1e-7
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    if lrate > lr_min:
        return lrate
    else:
        return lr_min

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.losses = []
        self.lr = []
        loss_out = 'loss.txt'
        outfile = open(loss_out,'w')
        outfile.write('Epoch Loss Val_Loss LR\n')
        outfile.close()
        
    def on_epoch_end(self, epoch, logs):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr: %.3e'%step_decay(len(self.losses)))
        loss_out = 'loss.txt'
        outfile = open(loss_out,'a+')
        outfile.write('%d %.3e %.3e %.3e\n'%(int(epoch),float(logs['loss']),float(logs['val_loss']),step_decay(epoch)))
        outfile.close()

def output(text):
    # output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.write('\n')
    newfile.close()

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    tf.keras.backend.set_floatx('float64')

    M = args.data_size
    dPCA = args.dPCA
    dh = args.dh
    arch = args.arch
    model = args.traj
    reg = args.reg
    load = args.load
    ae_model = args.ae_model
    ae_optim = args.ae_optim
    ae_wtdc = args.ae_wtdc
    ae_nlin = args.ae_nlin

    h = pickle.load(open('../autoencoder/dPCA%d/%s_w%.2e_l%d/rmodel%d/encoded_data.p'%(dPCA,ae_optim,ae_wtdc,ae_nlin,ae_model),'rb'))

    data_dir = '../../BD_FFoRM/preprocessed/na200_nq_100_qlim_0.10_L1680.0_R34.0/training_data/'
    stress = pickle.load(open(data_dir+'aligned_stress_snaps_%d.p'%M,'rb'))
    Q,phase_t = pickle.load(open(data_dir+'Qphase_snaps_%d.p'%M,'rb'))
    grad,E = pickle.load(open(data_dir+'GdotE_snaps_%d.p'%M,'rb'))

    red_sym_ind = np.array([0,1,2,4,5,8])
    # E = E[:,red_sym_ind]
    # stress = stress[:,red_sym_ind]

    E = np.reshape(E,(E.shape[0],3,3))
    grad = np.reshape(grad,(grad.shape[0],3,3))
    Q = np.reshape(Q,(Q.shape[0],3,3))
    QT = np.transpose(Q,axes=[0,2,1])
    aTEa = np.zeros(E.shape)
    for i in range(E.shape[0]):
        thetaQ = np.arctan2(Q[i,0,1],Q[i,0,0])
        phaseQ = -phase_t[i]/2.0 + thetaQ
        RQ = np.array([[np.cos(phaseQ),-np.sin(phaseQ),0.0],[np.sin(phaseQ),np.cos(phaseQ),0.0],[0.0,0.0,0.0]])
        aTEa[i] = np.transpose(RQ)@E[i]@RQ

    aTEa = np.reshape(aTEa,(aTEa.shape[0],9))
    Q = np.reshape(Q,(Q.shape[0],9))
    E = np.reshape(E,(E.shape[0],9))

    hE = np.concatenate((h,aTEa),axis=1)

    print('ROM, gradient, ROM_grad, stress shapes')
    print(h.shape,E.shape,hE.shape,stress.shape)

    wdir = 'dPCA%d/rdh%d/arch%d_reg%.6f/model%d'%(dPCA,dh,arch,reg,model)
    # wdir = 'l%d_n%d_tsp%d_dd%d_nd%d'%(nl,neurons,t_sp,dd,nd)

    if os.path.exists(wdir) == 0:
        os.makedirs(wdir,exist_ok=False)

    os.chdir(wdir)

    output('%s'%os.getcwd())

    ###########################################################################
    # Prepare for NN
    ###########################################################################

    # Some stress indices are NaN, not sure why
    # Probably related to block averaging uu,uuuu and Gd in BD_FFoRM/stress.py
    # Come back to resolve this in preprocessing later
    # For now just exclude snapshots where stress is NaN (~1k out of ~100k)
    print(stress.shape,hE.shape)
    ind_nan = np.argwhere(np.isnan(stress[:,0]))[:,0]

    print(ind_nan.shape)
    # for i in ind_nan:
        # print(i,stress[i])

    stress = np.delete(stress,ind_nan,axis=0)
    hE = np.delete(hE,ind_nan,axis=0)
    print(stress.shape,hE.shape)
    # exit()

    # Think about if this is necessary - should not be for Iq PCA modes
    # Stress magnitude can be large, may help to normalize
    # # Normalize data by subtracting off the mean and dividing by the standard deviation
    # stress_mean = np.mean(stress,axis=0)
    # stress_std = np.std(stress,axis=0)
    # stress = (stress - stress_mean[np.newaxis,:])/stress_std[np.newaxis,:]
    stress_mean = np.mean(stress)
    stress_std = np.std(stress)
    # stress = (stress - stress_mean)/stress_std
    print(stress_mean)
    print(stress_std)

    hE, stress = shuffle(hE, stress)
    
    # Split training and test data
    frac=0.8
    hE_train = hE[:round(M*frac),:]
    hE_test = hE[round(M*frac):M,:]

    stress_train = stress[:round(M*frac)]
    stress_test = stress[round(M*frac):M]

    ###########################################################################
    # Neural Networks
    ###########################################################################

    val_split = 0.2
    init_epoch = 0
    # Build the model
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(hE.shape[1])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128,activation='relu',kernel_regularizer=regularizers.l1(reg)))
    model.add(keras.layers.Dense(units=128,activation='relu',kernel_regularizer=regularizers.l1(reg)))
    model.add(keras.layers.Dense(stress.shape[1]))

    model.compile(loss='mse', optimizer='adam')

    # Save the architecture summary for quick viewing
    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    ###########################################################################
    # Load previously trained model
    ###########################################################################

    if load==1:
        if os.path.exists('model.h5'):
            model.load_weights('model.h5')
        else:
            model.load_weights('model_chpt.h5')

    ###########################################################################
    # Load previously trained model
    ###########################################################################

    elif load==2:
        model = tf.keras.models.load_model('model_chpt.h5')
        iterations = model.optimizer.iterations.numpy()
        steps_per_epoch = int(M*frac*(1.0-val_split)/32)
        init_epoch = int(iterations/steps_per_epoch)

    ###########################################################################
    # Train the model
    ###########################################################################

    if load==0 or load==2:
        MAX_EPOCHS = 500
        loss_history = LossHistory()
        lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model_chpt.h5')
        callbacks_list = [loss_history, lrate, checkpoint]
        history = model.fit(
            hE_train, stress_train, 
            epochs=MAX_EPOCHS, 
            # batch_size=16, 
            validation_split=val_split, 
            callbacks=callbacks_list,
            initial_epoch = init_epoch,
            verbose=2, 
            shuffle=True
        )

        # plot learning rate
        plt.figure()
        plt.plot(range(1,MAX_EPOCHS+1),loss_history.lr,label='learning rate')
        plt.xlabel("epoch")
        plt.xlim([1,MAX_EPOCHS+1])
        plt.ylabel("learning rate")
        plt.legend(loc=0)
        plt.grid(True)
        plt.title("Learning rate")
        # plt.show()
        plt.savefig('learning_rate.png')
        # plt.close(fig)
        
        # Save history for comparing models
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        # Plot MAE and MSE vs epochs
        plot_history(hist)
        plt.tight_layout()
        plt.savefig(open('Error.png','wb'))
        plt.close()

        # Save models
        model.save_weights('model.h5')

    ###########################################################################

    # NN prediction
    stress_pred = model.predict(hE_test)

    output('NN output shape %s'%(stress_pred.shape,))

    # x_test = np.transpose(x_test)
    # xhat = np.transpose(xhat)

    error = stress_test.flatten() - stress_pred.flatten()
    print('max error',max(error))

    # Plot error histogram
    plt.figure()
    # MSEshift=plot_hist(error,'NN',xmax=max(error))
    MSEshift=plot_hist(error,'NN')
    plt.tight_layout() 
    plt.savefig(open('Statistics.png','wb'))
    plt.close()

    mse_test = np.mean(error**2)
    print(mse_test)
    output('MSE test - %e'%mse_test)

    error = stress_test - stress_pred
    mse = np.mean(error**2,axis=0)
    print('MSE for individual components')
    # print(mse.shape)
    for i in range(len(mse)):
        print(i,mse[i])

    # start = 33000
    # maxt = 200
    # t = np.arange(0,maxt)*0.01

    # stress_pred_seq = model.predict(aE[start:start+maxt])

    # for i in range(maxt):
    #     # print(i,stress[i],stress_pred_seq[i])
    #     print(i,aE[start+i,0],aE[start+i,-9],stress[i],stress_pred_seq[i])

    # fig = plt.figure(figsize=(4,3),dpi=300)
    # ax = plt.subplot(111)
    # ax.plot(t,stress[start:start+maxt],'s',markersize=1,label='Data')
    # ax.plot(t,stress_pred_seq,'o',markersize=1,label='Prediction')
    # # ax.plot(t,E[:,0],label='E_{00}')
    # plt.legend()
    # # plt.title('MSE %.3e'%mse)
    # ax.set_ylabel(r'$\sigma_{%d}$'%component)
    # ax.set_xlabel(r'$t$')
    # plt.tight_layout()
    # plt.show()

    # exit()

    # stress = (stress - stress_mean[np.newaxis,:])/stress_std[np.newaxis,:]

    stress_mean = np.mean(stress,axis=0)
    stress_std = np.std(stress,axis=0)
    print(stress_mean)
    print(stress_std)
    stress_test = (stress_test - stress_mean[np.newaxis,:])/stress_std[np.newaxis,:]
    stress_pred = (stress_pred - stress_mean[np.newaxis,:])/stress_std[np.newaxis,:]

    # error = stress_test.flatten() - stress_pred.flatten()
    # mse_test = np.mean(error**2)
    # print(mse_test)
    # output('MSE test - %e'%mse_test)

    error = stress_test - stress_pred
    mse = np.mean(error**2,axis=0)
    print('Normalized MSE for individual components')
    for i in range(len(mse)):
        print(i,mse[i])

    pickle.dump(mse,open('norm_error.p','wb'))

    fig = plt.figure(figsize=(4,3),dpi=300)
    ax = plt.subplot(111)
    plt.semilogy(np.arange(6),mse[red_sym_ind],'o')
    # plt.title('MSE %.3e'%mse)
    plt.xticks([0,1, 2, 3, 4, 5], ['xx', 'xy', 'xz','yy','yz','zz'])
    ax.set_ylabel(r'$|\bar{\sigma}_{ij} - \bar{\tilde{\sigma}}_{ij}|^2$')
    # ax.set_xlabel(r'$$')
    plt.tight_layout()
    plt.savefig(open('normalized_error.png','wb'))
    # plt.show()

    # ind_max = np.argmax(np.mean(error**2,axis=1))
    # Emax = aE_test[ind_max,-9:]
    # # Emag = np.einsum('ijk,ijk->i',Emax,Emax)**0.5
    # print(ind_max)
    # for i in range(9):
    #     print(i,Emax[i],stress_pred[ind_max,i],stress_test[ind_max,i])


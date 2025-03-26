#! /home/cyoung37/anaconda3/bin/python3
import sys
# sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from torchinfo import summary
import parse
import scipy
from scipy import integrate
from scipy.interpolate import interp1d

# For importing the autoencoder
# sys.path.insert(0, '../')
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--dPCA', type=int, default=50) # Number of PCA modes retained
parser.add_argument('--dh', type=int, default=5) # Autoencoder latent dimension
parser.add_argument('--data_size', type=int, default=95181)  #IC from the simulation
parser.add_argument('--dt',type=float,default=0.01)
parser.add_argument('--batch_time', type=int, default=10)   #Samples a batch covers (this is 10 snaps in a row in data_size)
parser.add_argument('--batch_size', type=int, default=100)   #Number of IC to calc gradient with each iteration
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=2000)       #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)    #Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--arch', type=int, default=0)
parser.add_argument('--traj', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--strm', type=int, default=0)
parser.add_argument('--dir', type=str, default='') # Name of directory - use for code changes
parser.add_argument('--ae_model', type=int, default=0)
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

# Determines what solver to use
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    # This is the default
    from torchdiffeq import odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS during training
class ODEFunc(nn.Module):
    def __init__(self,dh,alpha):
        super(ODEFunc, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh+1),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(dh+6, 256),
        #     nn.ReLU(),
        #     nn.Linear(256,256),
        #     nn.ReLU(),
        #     nn.Linear(256, dh),
        # )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # E = Efunc(t.detach().numpy())
        h = y[0]
        t0 = y[1]
        # print(E.shape)
        # print(h.shape)
        # print(t0.shape)
        # exit()

        g = QTEQfunc(t.detach().numpy() + t0.detach().numpy())
        g = torch.tensor(g)
        g = g[:,np.newaxis,:]
        g = g.type(torch.FloatTensor)
        hg = torch.cat((h,g),axis=-1)
        # print(g.shape,hg.shape)
        # exit()
        return tuple([self.net(hg) - self.alpha*h] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]])

class ODE2(nn.Module):
    def __init__(self,dh,alpha):
        super(ODE2, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+9, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh),
        )
        self.netornt = nn.Sequential(
            nn.Linear(dh+9, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        for m in self.netornt.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # E = Efunc(t.detach().numpy())
        h = y[0]
        # print(h.shape)
        ornt = h[:,:,-1]
        h = h[:,:,:-1]
        t0 = y[1]
        # print(h.shape,ornt.shape,t0.shape)

        g = gfunc(t.detach().numpy() + t0.detach().numpy())
        g = torch.tensor(g)
        g = g[:,np.newaxis,:]
        g = g.type(torch.FloatTensor)
        hg = torch.cat((h,g),axis=-1)
        # exit()

        hRHS = self.net(hg) - self.alpha*h
        thetaRHS = self.netornt(hg) - self.alpha*ornt[:,:,np.newaxis]
        # print(hRHS.shape,thetaRHS.shape)
        RHS = torch.cat((hRHS,thetaRHS),axis=-1)
        # print(RHS.shape)
        # exit()    
        # return tuple([self.net(hg) - self.alpha*h] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]])
        # return RHS

        return tuple([RHS] + [torch.zeros_like(s_).requires_grad_(True) for s_ in y[1:]])


# Use the trained model with SciPy integrators to reduce cost of storing the gradients
class SciPyODE(nn.Module):
    def __init__(self,dh,alpha):
        super(SciPyODE, self).__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(dh+10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dh+1),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(dh+6, 64),
        #     nn.ReLU(),
        #     nn.Linear(64,64),
        #     nn.ReLU(),
        #     nn.Linear(64, dh),
        # )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # This is the evolution with the NN
        # g = QTEQfunc(t)
        g = gfunc(t)
        hg = np.concatenate((y,g))
        hg = torch.tensor(hg)
        hg = hg.type(torch.FloatTensor)
        return self.net(hg).detach().numpy() - self.alpha*y

        # E = Efunc(t)
        # hE = np.concatenate((y,E))
        # hE = torch.tensor(hE)
        # hE = hE.type(torch.FloatTensor)
        # return self.net(hE).detach().numpy() - self.alpha*y

# This class is used for updating the gradient
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

###############################################################################
# Functions
###############################################################################
# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(t,true_y,ic_indices):
    # s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    s = torch.from_numpy(np.random.choice(ic_indices, args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_t0 = t[s]
    # batch_t = torch.stack([t[s + i] for i in range(args.batch_time)], dim=0)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # return batch_y0, batch_t, batch_y
    return batch_y0, batch_t0, batch_t, batch_y

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

def best_path(path):

    text=open(path+'Trials.txt','r')
    MSE=[]
    for line in text:
        vals=line.split()
        # Check that the line begins with T, meaning it has trial info
        if vals[0][0]=='T':
            # Put MSE data together
            MSE.append(float(vals[1]))
    
    idx=np.argmin(np.asarray(MSE))+1

    return path+'Trial'+str(idx)

if __name__ == '__main__':

    batch_time = args.batch_time
    data_size = args.data_size
    dPCA = args.dPCA
    dh = args.dh
    arch = args.arch
    model = args.traj
    load = args.load
    dt = args.dt
    alpha = args.alpha
    # ae_model = 0
    ae_model = args.ae_model
    # args.data_size = args.data_size - args.dd*args.tsp

    # Load the data and put it in the proper form for training
    # dPCAin = 50
    data_dir = '../../BD_FFoRM/c_rods/training_output/'
    # anp = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/red_%d_PCA_modes_%d.p'%(dPCAin,data_size),'rb'))
    # anp = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/a_snaps_ma_clearnan_%d.p'%(data_size),'rb'))
    h = pickle.load(open('../autoencoder/dPCA%d/%s/rmodel%d/encoded_data.p'%(dPCA,args.dir,ae_model),'rb'))
    print(h.shape)
    Q,phase_raw = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/Qphase_snaps_%d.p'%data_size,'rb'))
    Q_ma,phase_t = pickle.load(open(data_dir+'na200_nq_100_qlim_0.10_L1680.0_R34.0/Qphase_ma_snaps_%d.p'%data_size,'rb'))
    grad = pickle.load(open(data_dir+'stress_L1680.0_R34.0/gradient_snapshots_clearnan_%d.p'%data_size,'rb'))
    E = pickle.load(open(data_dir+'stress_L1680.0_R34.0/E_snapshots_clearnan_%d.p'%data_size,'rb'))
    crossover_ind = pickle.load(open(data_dir+'crossover_indices_%d.p'%data_size,'rb'))

    # Create and enter working directory
    wdir = 'rdh%d/dPCA%d/%s/arch%d_tau%d_alpha%.1e/model%d'%(dh,dPCA,args.dir,arch,batch_time,alpha,model)
    if os.path.exists(wdir) == 0:
        os.makedirs(wdir,exist_ok=False)
    os.chdir(wdir)

    if load==0:
        output('%s'%os.getcwd())

    chpdir = 'chp'
    if(os.path.exists(chpdir)==0):
        os.makedirs(chpdir,exist_ok=False)

    # red_sym_ind = np.array([0,1,2,4,5,8])
    # # grad = grad[:,red_sym_ind]
    # E = E[:,red_sym_ind]
    [M,N] = h.shape

    # Exclude indices for initial conditions that crossover from traj to the next in the data
    ic_indices = np.arange(0,data_size)
    crossovers = np.zeros((len(crossover_ind),batch_time-1),dtype=int)
    for i in range(len(crossover_ind)):
        crossovers[i] = np.arange(crossover_ind[i] - (batch_time - 2),crossover_ind[i]+1)
        # print(i,crossovers[i])

    ic_indices = np.delete(ic_indices,crossovers.flatten())
    # for i in range(len(ic_indices)):
        # print(i,ic_indices[i])
    # exit()

    # Normalize phase data
    phase_mean = np.mean(phase_t)
    phase_std = np.std(phase_t)
    phase_t = (phase_t - phase_mean)/phase_std
    pickle.dump([phase_mean,phase_std],open('phase_stats.p','wb'))

    # Compute co-rotating gradient Q^T E Q
    E = np.reshape(E,(E.shape[0],3,3))
    Q = np.reshape(Q,(Q.shape[0],3,3))
    QT = np.transpose(Q,axes=[0,2,1])
    # QTEQ = QT@E@Q
    QTEQ = np.zeros(E.shape)
    for i in range(E.shape[0]):
        QTEQ[i] = QT[i]@E[i]@Q[i]
    QTEQ = np.reshape(QTEQ,(QTEQ.shape[0],9))

    # Append orientation to ROM for prediction with state
    print('ROM shape',h.shape)
    print('phase shape',phase_t.shape)
    print('QTEQ shape',QTEQ.shape)
    # ho = np.append(h,phase_t[:,np.newaxis],axis=1)
    ho = np.append(h,phase_t,axis=1)
    print('State shape',ho.shape)

    # Convert to torch tensor
    ho = torch.tensor(ho[:data_size,np.newaxis,:])
    ho = ho.type(torch.FloatTensor)
    tnp = np.arange(0,data_size)*dt
    gfunc = interp1d(tnp, grad, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    Efunc = interp1d(tnp, E, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    QTEQfunc = interp1d(tnp, QTEQ, axis = 0, kind = 'linear', bounds_error=False, fill_value="extrapolate")
    t = torch.tensor(tnp)
    t = t.type(torch.FloatTensor)

    lr_init = 1e-3
    if args.load==0 or args.load==2:
        it_per_drop = 500
        ###########################################################################
        # Initialize NN for learning the RHS and setup optimization parms
        ###########################################################################
        if args.load==0:
            # func = ODEFunc(dh,alpha)
            func = ODE2(dh,alpha)
            itr_start = 1
        elif args.load==2:
            # func = torch.load('model.pt')
            func = torch.load('chp/model.pt')
            func.eval()
            fout = 'Out.txt'
            for line in reversed(list(open(fout))):
                print(line)
                [itr_start,loss_start,time_start]=parse.parse('Iter {0} | Total Loss {1} | Time {2}\n',line)
                itr_start = int(itr_start)
                break
            lr_init = lr_init*0.5**(int(itr_start/it_per_drop))

        print('lr_init %.3e'%lr_init)
        # optimizer = optim.Adam(func.parameters(), lr=lr_init) #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        optimizer = optim.Adam(func.parameters(), lr=lr_init, weight_decay = 1.0e-5) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=it_per_drop, gamma=0.5)
        end = time.time()
        time_meter = RunningAverageMeter(0.97)
        loss_meter = RunningAverageMeter(0.97)

        # model_stats = summary(func, [(1,1),(1,1,dPCA)])
        # model_str = str(model_stats)

        # with open('model_summary.txt', 'w') as f:
        #     # model.summary(print_fn=lambda x: f.write(x + '\n'))
        #     f.write(model_str)

        err=[]
        ii = 0
        ###########################################################################
        # Optimization iterations
        ###########################################################################
        # print(itr_start,args.niters)
        for itr in range(itr_start, args.niters + 1):

            # Get the batch and initialize the optimizer
            optimizer.zero_grad()
            batch_y0, batch_t0, batch_t, batch_y = get_batch(t,ho,ic_indices)
            if itr==1:
                output('Batch Time Units: '+str(batch_t.detach().numpy()[-1])+'\n')

            # Make a prediction and calculate the loss
            pred_y = odeint(func, tuple([batch_y0,batch_t0]), batch_t)
            loss = torch.mean((pred_y[0][1:] - batch_y[1:])**2)
            loss.backward() #Computes the gradient of the loss (w.r.t to the parameters of the network?)
            # Use the optimizer to update the model
            optimizer.step()
            # Update the learning rate
            scheduler.step()

            # Print out the Loss and the time the computation took
            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())
            if itr % args.test_freq == 0:
                # # Model checkpoint
                # torch.save({
                #         'epoch':itr,
                #         'model_state_dict':func.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'loss':loss.item()
                #     }, chp_path)
                if itr % 100 == 0:
                    # torch.save(func,'chp/epoch_%d_loss_%.3e.pt'%(itr,loss.item()))
                    # torch.save(func.state_dict(),'chp/sd_epoch_%d_loss_%.3e.pt'%(itr,loss.item()))
                    torch.save(func,'chp/model.pt')
                    torch.save(func.state_dict(),'chp/sd.pt')
                with torch.no_grad():
                    # print(itr,loss.item(),time.time() - end)
                    print('epoch {}, loss:{:.2e}, lr:{:.6f}, time {:.6f}'
                        .format(itr + 1, loss.item(), optimizer.param_groups[0]['lr'], time.time() - end))
                    err.append(loss.item())
                    output('Iter {:04d} | Total Loss {:.2e} | Time {:.6f} | lr {:.6f}'.format(itr, loss.item(),time.time() - end, optimizer.param_groups[0]['lr'])+'\n')
                    ii += 1
            end = time.time()
        
        ###########################################################################
        # Plot results and save the model
        ###########################################################################
        torch.save(func, 'model.pt')
        torch.save(func.state_dict(),'state_dict.pt')
        #pickle.dump(func,open('model.p','wb'))

        # Plot the learning
        plt.figure()
        # plt.semilogy(np.arange(args.test_freq,args.niters+1,args.test_freq),np.asarray(err),'.-')
        plt.semilogy(np.arange(itr_start,args.niters+1,args.test_freq),np.asarray(err),'.-')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.savefig('Error_v_Epochs.png')

    elif args.load==1:
        # func = torch.load('model.pt')
        func = torch.load('chp/model.pt')
        func.eval()

        # torch.save(func.state_dict(),'state_dict.pt')

    exit()

    func_sp = SciPyODE(dh,alpha)
    # func_sp.load_state_dict(torch.load('state_dict.pt'))
    func_sp.load_state_dict(torch.load('chp/sd.pt'))
    func_sp.eval()

    # start = 50400
    h = h.detach().numpy()
    print(h.shape)
    h = np.squeeze(h)
    print(h.shape)
    strm = args.strm
    start = crossover_ind[strm]+1
    end = crossover_ind[strm+1]+1
    print(start,end)
    nt = end - start
    tspan = tnp[start:start+nt]
    # gdat = g[start:end]
    # ginterp = gfunc(tspan)
    # print(ginterp.shape)
    sol = scipy.integrate.solve_ivp(func_sp,[tspan[0],tspan[-1]],h[start],t_eval=tspan,max_step=0.01)
    pred_y = np.transpose(sol['y'])

    # comp1 = 0
    # comp2 = 0
    # plt.figure(figsize=(4,3),dpi=300)
    # plt.plot(tspan, pred_y[:,comp1],'s',fillstyle='none',markersize=2,label='SciPyODE mode %d'%(comp1+1))
    # plt.plot(tspan, pred_y[:,comp2],'o',fillstyle='none',markersize=2,label='SciPyODE mode %d'%(comp2+1))
    # plt.plot(tspan, a[start:start+nt,0,comp1],'s',markersize=2,label='Data mode %d'%(comp1+1))
    # plt.plot(tspan, a[start:start+nt,0,comp2],'o',markersize=2,label='Data mode %d'%(comp2+1))
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('PCA mode')
    # plt.tight_layout()
    # plt.show()

    fig = plt.figure(figsize=(12,6),dpi=200)
    ax = plt.subplot(231)
    plt.plot(tspan,h[start:end,0],'o-',label='Data')
    plt.plot(tspan,pred_y[:,0],'s--',label='Prediction')
    plt.legend()
    plt.ylabel(r'$h_0$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(232)
    plt.plot(tspan,h[start:end,1],'o-')
    plt.plot(tspan,pred_y[:,1],'s--')
    plt.ylabel(r'$h_1$')
    plt.xlabel(r'$t$')

    ax = plt.subplot(233)
    plt.plot(tspan,h[start:end,2],'o-')
    plt.plot(tspan,pred_y[:,2],'s--')
    plt.ylabel(r'$h_2$')
    plt.xlabel(r'$t$')

    # ax = plt.subplot(234)
    # plt.plot(tspan,h[start:end,3],'o-')
    # plt.plot(tspan,pred_y[:,3],'s--')
    # plt.ylabel(r'$h_2$')
    # plt.xlabel(r'$t$')

    # ax = plt.subplot(235)
    # plt.plot(tspan,h[start:end,4],'o-')
    # plt.plot(tspan,pred_y[:,4],'s--')
    # plt.ylabel(r'$h_2$')
    # plt.xlabel(r'$t$')

    plt.tight_layout()

    # fig = plt.figure(figsize=(8,4),dpi=200)

    # # ax = plt.subplot(111)
    # ax = plt.subplot(231)
    # ax.plot(tspan,Edat[:,0],'s',markersize=1,label='Data')
    # ax.plot(tspan,Einterp[:,0],'o',markersize=1,label='Interp')
    # plt.legend()
    # ax.set_ylabel(r'$E_{xx}$')

    # ax = plt.subplot(232)
    # ax.plot(tspan,Edat[:,1],'s',markersize=1)
    # ax.plot(tspan,Einterp[:,1],'o',markersize=1)
    # ax.set_ylabel(r'$E_{xy}$')

    # ax = plt.subplot(233)
    # ax.plot(tspan,Edat[:,2],'s',markersize=1)
    # ax.plot(tspan,Einterp[:,2],'o',markersize=1)
    # ax.set_ylabel(r'$E_{xz}$')

    # ax = plt.subplot(234)
    # ax.plot(tspan,Edat[:,3],'s',markersize=1)
    # ax.plot(tspan,Einterp[:,3],'o',markersize=1)
    # ax.set_ylabel(r'$E_{yy}$')
    # ax.set_xlabel(r'$t$')

    # ax = plt.subplot(235)
    # ax.plot(tspan,Edat[:,4],'s',markersize=1)
    # ax.plot(tspan,Einterp[:,4],'o',markersize=1)
    # ax.set_ylabel(r'$E_{yz}$')
    # ax.set_xlabel(r'$t$')

    # ax = plt.subplot(236)
    # ax.plot(tspan,Edat[:,5],'s',markersize=1)
    # ax.plot(tspan,Einterp[:,5],'o',markersize=1)
    # ax.set_ylabel(r'$E_{zz}$')
    # ax.set_xlabel(r'$t$')

    # plt.tight_layout()

    plt.show()


# SIMPLE
Scattering Informed Microstructure Prediction during Lagrangian Evolution - Young et al. Rheo Acta 2023

# Instructions for training SIMPLE models

Last update - 07/25/2023 by CDY

# Notes on training data in BD_FFoRM #####

For simplicity of this tutorial and file size reduction, the data generation process, ....
preprocessing for frame indifference and phase alignment, and the linear dimension ...
reduction by PCA have been already performed.
The training data is stored in: 'BD_FFoRM/preprocessed/na200_nq_100_qlim_0.10_L1680.0_R34.0/training_data'
Individual FFoRM streamlines for testing the trained models are stored in:
'BD_FFoRM/preprocessed/na200_nq_100_qlim_0.10_L1680.0_R34.0/FFoRM_streamlines'
(note the string after preprocessed will vary depending on the parameters of the scattering model)

The key for file names to data in 'training_data' is:
M = no. of snapshots = 106663 in this tutorial
dPCA = no. of PCA modes retained = 50 in this tutorial
aligned_stress_snaps_M.p: snapshots of aligned stress tensor \sigma_P,\alpha \in M x 9
GdotE_snaps_M.p: lab-frame velocity gradient tensor \in M x 9 and rate of strain tensor \in M x 9
Note velocity gradient tensor is for visualizing flow type, even though it is not a model input
PCA_modes_M.p: The leading X PCA modes after truncation. shape d_o x 100
Note I kept 100 here just to test effect of dPCA=25,50,100. Use this matrix to reconstruct after time evolution/decoding
Qphase_snaps_M.p: snapshots of vorticity rotation matrix Q \in M x 9 and co-rotating phase \alpha \in M x 1

For files in FFoRM_streamlines, there is on file for each streamline
FRR = Q1/Q2 determines the flow type parameter
Q2 determines the flow strength
strm determines the initial position of the Lagrangian streamline in the FFoRM
IQphase_...p contains the output of Pq_rot_l2c.py for each streamline in the order:
IQa,Q,phase_t_corot,Gdotout,Eout,QaT_stress_Qa
IQa: full scattering observation after account for co-rotation and phase alignment
Q: vorticity rotation vector
phase_t_corot: corotating phase \alpha
Gdotout: lab frame velocity gradient tensor interpolated on even intervals \Delta t = 0.01
Eout: lab frame rate of strain tensor "
QaT_stress_Qa: phase-aligned stress tensor

This is all the data that should be required for training AEs, NODEs, and stress models and test on a few streamlines
For completeness, the files to generate all data from scratch are included in 'openfoam' and 'BD_FFoRM'
Further details on the complete data generation process are included after instructions for training (in progress)
Data for testing trained models on data not seen in the FFoRM are in BD_test_data (instructions in progress)

# Training summary

1) Train autoencoder using IRMAE.py
2) Determine number of non-zero singular values
3) Rerun IRMAE.py with --train 0 and truncate to --dh <no. of non-zero SVs>
4) Train NODE with NODE.py
5) Train stress model with stress_haligned.py
6) Deploy trained SIMPLE framework with pred_interp.py or pred_extrap.py

# Training autoencoders

Enter the 'autoencoder' directory
IRMAE.py is the code to train the autoencoder for input phase-aligned PCA modes
See the file for comments on the inputs
create_jobs.py will create a sweep over NN hyperparameters
The goal is to obtain a model with a sharp drop in the singular values with a low MSE
Increasing the number of linear layers and weight decay strength yields fewer nonzero singular values (good) ...
but a higher MSE (bad)
Usually some balance can be found where there is a sharp drop in the SV but a low loss

To train an individual autoencoder on desktop using default parameters:
$ python IRMAE.py
To vary inputs:
$ python IRMAE.py --train 0 --model 1 --wtdc 5e-3 --nlin 4 --optim AdamW

After training is done, inspect the SV spectrum to determine the manifold dimension
Then, run again and specify dh to truncate the latent space after SVD:
$ python IRMAE.py --train 0 --model 1 --wtdc 5e-3 --nlin 4 --optim AdamW --dh 6

Now the encoded training data will be saved in 'encoded_data.p', which can be input to ...
the time evolution NODE or the stress NN

# Training Neural ODEs

Enter the 'integrate' directory
NODE.py is the code to train the evolution of the latent space phase-aligned structure (h) ...
and the co-rotating phase \alpha
See the code for comments on the inputs
Some hyperparameter tuning of forecast time, network architecture, etc. may be necessary
The last set of parameters I tested produced consistent extrapolation conditions (see below) ...
for 6-7 of 10 models with relatively arbitrary parameter selection

$ python NODE.py --ae_model 1

# Training stress models

Enter the 'stress' directory
stress_haligned.py is the code to train the model mapping the phase-aligned latent space and rate of strain tensor ...
to the phase-aligned stress tensor
Note to obtain the lab frame stress tensor, you need to apply the rotation matrix
When predicting the microstructure and phase in time, you will use the predicted phase to perform this rotation
Be careful about normalizing the stress tensor inputs to the NN - the xz and yz component magnitudes are usually small ...
because the flow is planar

$ python stress_haligned.py --ae_model 1

# Deploying the fully trained framework

After training the autoencoder, NODE, and stress model, the three can be used together
Under the integrate directory, pred_interp.py will predict FFoRM streamlines, and pred_extrap.py will predict ...
arbitrary flow protocols contained within BD_test_data
In both of these codes, the scattering IC is phase-aligned, encoded, and forecasted in time using the known ...
rate of strain tensor. The predicted latent space and phase can be used to predict the stress as well
For visual comparison, the predicted latent space can be decoded and rotated back to the lab frame. The optional ...
movie input controls whether to generate a .mp4 of the lab frame time evolution

$ python pred_interp.py --dh 6 --movie 1 --Q2 8.9

# Generating training data - further details

UNDER CONSTRUCTION


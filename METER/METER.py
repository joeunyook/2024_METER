import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.nn.init as init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import argparse
import scipy.io
from hypnettorch.mnets import MLP #multilayer perceptron from the hypenettorch library
from hypnettorch.hnets import HMLP#hypernetwork multi layer perceptron
import warnings
import matplotlib.pyplot as plt
import math
from edl_pytorch import Dirichlet, evidential_classification #FOR EVIDENTIAL DEEP LEARNING

class METER(nn.Module):  #nn.Module is a base class for all neural network modules       
    def __init__(self, args, in_dim, params, alpha, N,  lr, hlr, device, emb_rate=2, 
                 encoder_layer=10, decoder_layer=10, hyperenc_layer=100, hyperdec_layer=100):
        super(METER, self).__init__()
        self.params = params #stores parameters for the model
        self.in_dim = in_dim #input dimension
        self.out_dim = int(in_dim*emb_rate)#output dimension
        self.init_data = torch.randn(N, self.in_dim).to(device) #randomly initialized data of shape (N,in_dim) 
        self.init_data.requires_grad = False #prevent gradients from being computed for init_data
        self.mean = 0 #mean
        self.std = 0 #standard deviation
        self.encoder_layer = encoder_layer # number of layers for encoder and decoders 
        self.decoder_layer = decoder_layer
        self.hyperenc_layer = hyperenc_layer
        self.hyperdec_layer = hyperdec_layer
        '''Static'''
        self.encoder_static = MLP(self.in_dim, self.out_dim, hidden_layers=[self.encoder_layer,self.encoder_layer], 
                                  no_weights=False)   #static autoencoder using a MLP(Multilayer perceptron)  
        self.decoder_static = MLP(self.out_dim, self.in_dim, hidden_layers=[self.decoder_layer,self.decoder_layer], 
                                  no_weights=False) #static decoder using MLP 
        '''dynamic'''
        self.encoder = MLP(self.in_dim, self.out_dim, hidden_layers=[self.encoder_layer,self.encoder_layer], 
                            no_weights=True)    #dynamic encoder and decoder with the no_weights parameter set to true
        self.decoder = MLP(self.out_dim, self.in_dim, hidden_layers=[self.decoder_layer,self.decoder_layer], 
                            no_weights=True)
        self.hyperen = HMLP(target_shapes=self.encoder.param_shapes,cond_in_size=in_dim,
                            layers=(self.hyperenc_layer,self.hyperenc_layer) ) #hypernetworks to generate dynamic weights 
                                                                                #for the encoder and decoder
        self.hyperde = HMLP(target_shapes=self.decoder.param_shapes,cond_in_size=in_dim,
                            layers=(self.hyperdec_layer,self.hyperdec_layer) )
        

        self.edl_model = nn.Sequential(nn.Linear(self.in_dim, self.in_dim*2),  # two input dim
                                       nn.ReLU(),
                                       Dirichlet(self.in_dim*2, 2),  # two output classes
                                       )
        # Sequential model with a linear layer, RELU(Rectified linear unit) activation, and a Dirichlet layer for evidential deep learning 
        
        
        
        self.alpha = alpha #smoothing factor
        self.clock = 0 #counter variable
        #self.hyperen.internal_params,self.hyperde.internal_params
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)     #Adam optimizers for the params and dynamic params
        self.optimizer_d = torch.optim.Adam(self.parameters(), lr=hlr) # for dynamic
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)#Learning rate schedulers for the optimizers
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=0.9) #for dynamic
        self.loss_fn = nn.MSELoss()  # MSE loss function without scalar
        self.loss_la = nn.MSELoss(reduce=False)  #vector
        self.count = 0
        # self.hyperen.conditional_params.requires_grad = False
        # self.hyperde.conditional_params.requires_grad = False
        self.args = args #command line arguments 
        self.device = device #device to run the model on
        
    def train_autoencoder(self, data, epochs=2000, mode='dynamic',  #function to train the autoencoder
                          static_encoder_weight=None,static_decoder_weight=None,thres_rate=0.05,lamb=0.001):
        self.mean, self.std = self.init_data.mean(0), self.init_data.std(0) #compute the mean and standard deviation of the initial data
        new = (data - self.mean) / self.std #normalize the input data
        new[:, self.std == 0] = 0
        new = Variable(new)
        loss_list = np.zeros((epochs)) #initialize a list to store losses for each epoch
        
        for epoch in range(epochs): #EPOCH loop - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
            losses_recon = [] #list to store reconstruction
            losses_edl = [] #list to store edl loss
            self.optimizer.zero_grad() #zero the gradients for both optimizer
            self.optimizer_d.zero_grad()
            
            '''
            FROM HERE, MODE SPECIFIC TRAINING
            '''
            if mode == 'dynamic':
                encoder_weight = self.hyperen.forward(cond_input=new.mean(0).view(1, -1)) # Generate the weights using hypernetwork
                decoder_weight = self.hyperde.forward(cond_input=new.mean(0).view(1, -1)) # Generate the weights 
                z,_ =self.encoder.forward(new + 0.001*torch.randn_like(new).to(self.device),weights=encoder_weight)
                output,_ = self.decoder.forward(z,weights=decoder_weight)  #Forward pass through the encoder and decoder with the generated weights

            elif mode == 'static':
                z,static_encoder_weight = self.encoder_static.forward(new + 0.001*torch.randn_like(new).to(self.device))
                output,static_decoder_weight = self.decoder_static.forward(z) #FOrward pass through the static encoder and decoder

            elif mode == 'hybrid':
                encoder_weight = self.hyperen.forward(cond_id=0) # Generate the weights 
                decoder_weight = self.hyperde.forward(cond_id=0) # Generate the weights  
                encoder_weight_hybrid = [] 
                for j in range(len(encoder_weight)):
                    encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
                decoder_weight_hybrid = [] #combine static and dynamic weights to form hybrid weights
                for j in range(len(decoder_weight)):
                    decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j])  

                z,_ =self.encoder.forward(new + 0.001*torch.randn_like(new).to(self.device),
                                        weights=encoder_weight_hybrid)
                output,_ = self.decoder.forward(z,weights=decoder_weight_hybrid) #Forward pass
                
            elif mode == 'hybrid+edl':
                encoder_weight = self.hyperen.forward(cond_input=new.mean(0).view(1, -1)) # Generate the hybrid weights 
                decoder_weight = self.hyperde.forward(cond_input=new.mean(0).view(1, -1)) # Generate the hybrid weights 
                encoder_weight_hybrid = [] 
                for j in range(len(encoder_weight)): #combine generated weights with static weights for hybrid weights
                    encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
                decoder_weight_hybrid = [] 
                for j in range(len(decoder_weight)):
                    decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j])  
                
                z,_ =self.encoder.forward(new + 0.001*torch.randn_like(new).to(self.device), #new is a input data
                                        weights=encoder_weight_hybrid) #forward pass through the encoder 
                output,_ = self.decoder.forward(z,weights=decoder_weight_hybrid) #forward pass through decoder
                #perform edl detection by computing static losses and determining thresholds
                static_z,_ = self.encoder.forward(new.to(self.device), weights=static_encoder_weight)
                static_output,_ = self.decoder.forward(static_z,weights=static_decoder_weight)
                total_static_loss = self.loss_la(static_output, new).mean(-1) #static loss
                
                #determine the threshold based on the static loss
                thres = total_static_loss.reshape(-1,1).sort(0,True)[0][int(len(total_static_loss.reshape(-1,1))*thres_rate)]
                # print('output.shape:', output.shape)
                if epoch <= 100:
                    #for the first 100 epochs fake labels are generated based on whether static loss exceed the threshold
                    # create fake labels for initial training epochs
                    fake_label = torch.zeros((output.shape[0],output.shape[1],1))  #,output.shape[2]
                    fake_label = torch.from_numpy(np.array(np.where(total_static_loss>thres, 1, 0),dtype=np.int64))
                    
                    #compute the dirichlet prediction
                    pred_dirchlet = self.edl_model(static_output) # new
                    loss_edl = evidential_classification(pred_dirchlet, fake_label, lamb=lamb) # compute EDL LOSS with regularization -> see loss.py
                    # print('epoch <= 100, labels anomaly rate', sum(fake_label)/len(fake_label)) 
                    
                else: #for later than 100 epochs, compute uncertainty metric
                    pred_dirchlet = self.edl_model(static_output)
                    total_pred_dirchlet = pred_dirchlet.sum(-1, keepdims=True)
                    expected_p = pred_dirchlet / total_pred_dirchlet
                    eps = 1e-7
                    point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
                    data_uncertainty = torch.sum((pred_dirchlet/ total_pred_dirchlet) * 
                                     (torch.digamma(total_pred_dirchlet + 1) - torch.digamma(pred_dirchlet + 1)), dim=1)
                    
                    mu_e = self.args.mu_e #determine the uncertainty threshold
                    thres_e = data_uncertainty.reshape(-1,1).sort(0,True)[0][int(len(data_uncertainty.reshape(-1,1))*mu_e)]

                    #create fake labels based on the uncertainty threshold
                    fake_label = torch.zeros((static_output[data_uncertainty<=thres_e].shape[0],output.shape[1],1))  #,output.shape[2]
                    fake_label = torch.from_numpy(np.array(np.where(total_static_loss[data_uncertainty<=thres_e]>thres, 1, 0),dtype=np.int64))
                    pred_dirchlet = self.edl_model(static_output[data_uncertainty<=thres_e]) # compute Dirichlet prediction for uncertain samples
                    loss_edl = evidential_classification(pred_dirchlet, fake_label, lamb=lamb) # compute edl loss again with regularization
                    # print('labels anomaly rate', sum(fake_label)/len(fake_label))  
                    
            else:
                raise Exception('wrong mode setting')
            
            '''Loss combination and back propagation'''
            
            loss_recon = self.loss_fn(output, new) 
            if mode == 'hybrid+edl': #for hybrid edl mode, total loss includes reconsruction loss and EDL loss weighted by  self.args.beta_e * loss_edl
                # loss = loss_recon + 0.3 * loss_edl    #beta_e *
                loss = loss_recon + self.args.beta_e * loss_edl
                losses_edl.append(loss_edl.cpu().detach().numpy())
            else:
                loss = loss_recon 
            losses_recon.append(loss_recon.cpu().detach().numpy())
            loss.backward()   #retain_graph=True / Gradients are computes using backpropagation
            if mode == 'static': #optimizers update model params
                self.optimizer.step()
            else:
                self.optimizer_d.step() 

        '''post_epoch operations'''
        loss_list[epoch] = loss

        
        if mode == 'dynamic': #forward pass is performed to compute the latent representaions Z_all based on the mode 
            z_all,_ = self.encoder.forward(new.to(self.device),weights=encoder_weight)
        elif mode == 'static':
            z_all,_ = self.encoder_static.forward(new.to(self.device))
        elif mode in ['hybrid' , 'hybrid+edl']:
            z_all,_ = self.encoder.forward(new.to(self.device),weights=encoder_weight_hybrid)
            
        self.z_mean, self.z_std = z_all.mean(0), z_all.std(0) #update mean and standard deviation of the latent representation
        # print('z:', self.z_mean.shape, self.z_std.shape, z_all.shape,self.z_memory.shape)                
        if mode == 'dynamic':
            return encoder_weight, decoder_weight
        elif mode == 'static':
            return static_encoder_weight, static_decoder_weight
        elif mode in ['hybrid' , 'hybrid+edl']:
            return encoder_weight_hybrid, decoder_weight_hybrid
        
        
        '''
        Forward Method
        '''
            

    def forward(self, x, static_encoder_weight=None, static_decoder_weight=None):
        new = (x - self.mean) / self.std #input X is normalized
        new[:, self.std == 0] = 0  #zero the columns ehre std = 0 
        encoder_weight = self.hyperen.forward(cond_input=new.mean(0).view(1, -1)) # Generate the weights for dynamic encoder
        decoder_weight = self.hyperde.forward(cond_input=new.mean(0).view(1, -1)) # Generate the weights for dynamic decoder
        expected_p = None #variables for expected probability, data uncertainty, distributuional uncertainty are initialized
        data_uncertainty = None 
        distributional_uncertainty = None
        use_dynamic = 0
        #MODE SPECIFIC FORWARD PASS
        if self.args.mode == 'dynamic': #-> forward pass using dynamic weights
            z_emb,_ = self.encoder.forward(new,weights=encoder_weight)
            output,_ = self.decoder.forward(z_emb,weights=decoder_weight)
        elif self.args.mode == 'static': #-> forward pass using static weights
            z_emb,_ = self.encoder_static.forward(new)
            output,_ = self.decoder_static.forward(z_emb)
        elif self.args.mode == 'hybrid': #conmbine static and dynamic weights to form hybrid weights and then perform forward pass
            encoder_weight_hybrid = [] 
            for j in range(len(encoder_weight)):
                encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
            decoder_weight_hybrid = [] 
            for j in range(len(decoder_weight)):
                decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j]) 
            z_emb,_ = self.encoder.forward(new,weights=encoder_weight_hybrid)
            output,_ = self.decoder.forward(z_emb,weights=decoder_weight_hybrid)
        elif self.args.mode == 'hybrid+edl': #combine static+dynamic weight to form hybrid weights
            encoder_weight_hybrid = [] 
            for j in range(len(encoder_weight)):
                encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
            decoder_weight_hybrid = [] 
            for j in range(len(decoder_weight)):
                decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j]) 
            #forward pass through static encoder and decoder
            z_emb, _ = self.encoder_static.forward(new)
            output, _ = self.decoder_static.forward(z_emb)
            pred_sd = self.edl_model(output)
            total_pred_sd = pred_sd.sum(-1, keepdims=True)
            expected_p = pred_sd / total_pred_sd
            eps = 1e-7
            point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
            data_uncertainty = torch.sum((pred_sd / total_pred_sd) * 
                                         (torch.digamma(total_pred_sd + 1) - torch.digamma(pred_sd + 1)), dim=1)
            distributional_uncertainty = point_entropy - data_uncertainty
            distributional_uncertainty_threshold = self.args.uncertainty_threshold
            
            #Switch to dynamic if needed
            if expected_p[0][0].round() == 0 or distributional_uncertainty>distributional_uncertainty_threshold:   #dynamic
                use_dynamic = 1
                self.count = self.count + 1
                z_emb,_ = self.encoder.forward(new,weights=encoder_weight_hybrid)
                output,_ = self.decoder.forward(z_emb,weights=decoder_weight_hybrid)
        else:
            raise Exception('wrong mode setting')

#loss calculation
        loss_values = torch.norm(new - output,  p=1) 
        score = loss_values #+ lam*chis
        if expected_p == None:
            return score ,self.count, expected_p, None, data_uncertainty , distributional_uncertainty,use_dynamic
        else:
            return score ,self.count, expected_p, expected_p[0][0].round(), data_uncertainty , distributional_uncertainty,use_dynamic
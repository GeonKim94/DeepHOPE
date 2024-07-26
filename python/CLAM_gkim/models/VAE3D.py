import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

import h5py

# Define the VAE architecture
class VAE3D(nn.Module):
    def __init__(self, input_channels=1, latent_dim=1024):
        super(VAE3D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential()
            #384x384x32 x16

        self.encoder.add_module('conv1',nn.Conv3d(input_channels, 16, kernel_size=(7, 7, 1), stride=(3, 3, 1), padding=(3, 3, 0), padding_mode = 'replicate'))
        self.encoder.add_module('act1',nn.ReLU())
            #128x128x32 x16

        self.encoder.add_module('conv2',nn.Conv3d(16, 32, kernel_size=(5, 5, 1), stride=(2, 2, 1), padding=(2, 2, 0), padding_mode = 'replicate'))
        self.encoder.add_module('act2',nn.ReLU())
            #64x64x32 x32

        self.encoder.add_module('conv3',nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), padding_mode = 'replicate'))
        self.encoder.add_module('act3',nn.ReLU())
            #32x32x16 x64

        self.encoder.add_module('conv4',nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), padding_mode = 'replicate'))
        self.encoder.add_module('act4',nn.ReLU())
            #16x16x8 x64

        self.encoder.add_module('conv5',nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), padding_mode = 'replicate'))
        self.encoder.add_module('act5',nn.ReLU())
            #8x8x4 x64

        self.fc_mu = nn.Linear(8*8*4*64, latent_dim)
        self.fc_logvar = nn.Linear(8*8*4*64, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim,8*8*4*64)
        self.act_dec = nn.ReLU()

        self.decoder = nn.Sequential()
            #8x8x4 x64
        self.decoder.add_module('convt1',nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2)))
        self.decoder.add_module('act1',nn.ReLU())
        #need crop in forward
            #16x16x8 x64

        self.decoder.add_module('convt2',nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2)))
        self.decoder.add_module('act2',nn.ReLU())
        #need crop in forward
            #32x32x16 x64

        self.decoder.add_module('convt3',nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2)))
        self.decoder.add_module('act3',nn.ReLU())
        #need crop in forward
            #64x64x32 x32
            
        self.decoder.add_module('convt4',nn.ConvTranspose3d(32, 16, kernel_size=(5, 5, 1), stride=(2, 2, 1)))
        self.decoder.add_module('act4',nn.ReLU())
        #need crop in forward
            #128x128x32 x16
            
        self.decoder.add_module('convt5',nn.ConvTranspose3d(16, input_channels, kernel_size=(7, 7, 1), stride=(3, 3, 1)))
        self.decoder.add_module('act5',nn.Sigmoid())
        #need crop in forward
            #384x384x32 x1
    
    def encode(self, x):
        for module_encoder in self.encoder:
            x = module_encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        z = self.fc_dec(z)
        z = self.act_dec(z)
        z = z.view(z.size(0), 64, 8, 8, 4)
        
        for module_decoder in self.decoder:
            shape_prev = z.shape
            z = module_decoder(z)
            if isinstance(module_decoder, torch.nn.ConvTranspose3d):
                
                shape_want = tuple(a*b for a, b in zip(shape_prev[2:], module_decoder.stride))
                margin = tuple(a-b for a, b in zip(z.shape[2:], shape_want))
                margin_start = tuple(math.floor(a/2) for a in margin)
                margin_end = tuple(b-a for a, b in zip(margin, margin_start))
                margin_end = list(margin_end)
                for i in range(len(margin_end)):
                    if margin_end[i] == 0:
                        margin_end[i] = None
                margin_end = tuple(margin_end)

                z = z[:,:,margin_start[0]:margin_end[0],
                      margin_start[1]:margin_end[1],
                      margin_start[2]:margin_end[2]]
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# Define the VAE loss function # not used by G Kim (BCE is for binary image)
def vae_loss(reconstruction, x, mu, logvar):
    BCE = nn.BCELoss(reduction='mean')(reconstruction, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(BCE.item())
    print(KLD.item())
    return BCE+KLD

# Define the MSE loss function
def mse_loss(recon_x, x):
    return nn.MSELoss(reduction='mean')(recon_x, x)

# Define the KLD loss function
def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
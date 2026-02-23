##################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: Unet Architecture
# 
##################################################################

# Import libraries 
import torch
import torch.nn as nn

# Import functions
from dmfco2.unet_blocks import Unet_DownBlock, Unet_MidBlock, Unet_UpBlock
from dmfco2.time_embedding import get_time_embedding
from dmfco2.vqvae_architecture import VQVAE

# Define Conditional Unet
class Conditional_Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    def __init__(self, 
                 t_emb_dim, 
                 z_channels, 
                 image_cond, 
                 scalar_cond):
        
        super(Conditional_Unet, self).__init__()
        
        self.t_emb_dim = t_emb_dim
        self.z_channels = z_channels # Latent space channels 
        self.image_cond = image_cond # Enable 2D image conditioning
        self.scalar_cond = scalar_cond # Enable scalar conditioning
        
        im_channels = self.z_channels # The input channels are the latent channels and not the pixel space channels 
        self.down_channels = [64, 128, 256, 512]
        self.mid_channels = [512, 512, 256]
        self.t_emb_dim = 256
        self.image_cond = True
        self.scalar_cond = False
        self.down_sample = [True, True, True]
        self.num_down_layers = 1  # You can set it to 1 to decrease the architecture size 
        self.num_mid_layers = 1   # You can set it to 1 to decrease the architecture size 
        self.num_up_layers = 1    # You can set it to 1 to decrease the architecture size 
        self.norm_channels = 8    # they put 32 on the Unet but I used 8 
        self.attns = [True, True, True]
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # 2D Image Condition using **VQ-VAE Encoder ARCHITECTURE** (Not Pre-trained!)
        # We use also 2D Image Conditioning because all the conditions are mapped into 2D maps 
        if self.image_cond: 
            vqvae_encoder = VQVAE() # Create a new VQ-VAE encoder model
           
            # Manually redefine the first convolution layer to accept 12 channels
            self.cond_encoder_conv_in = nn.Conv2d(
                in_channels=12,  # Change from 1 → 12 channels for multi-conditioning
                out_channels=vqvae_encoder.down_channels[0], 
                kernel_size=3, 
                padding=1
            )
            self.cond_encoder_layers = vqvae_encoder.encoder_layers # Down blocks
            self.cond_encoder_mids = vqvae_encoder.encoder_mids # Mid blocks
            self.cond_encoder_norm_out = vqvae_encoder.encoder_norm_out
            self.cond_encoder_conv_out = vqvae_encoder.encoder_conv_out # Latent projection
            self.cond_encoder_pre_quant = vqvae_encoder.pre_quant_conv # Pre-quant projection
            self.cond_encoder_quantize = vqvae_encoder.quantize # Quantization function
            
            print("Image Channels:", im_channels)
            print("Condition Channels:", vqvae_encoder.z_channels) # which corresponds to the codebook size. From 12 channels to 128 channels to enhance the condition representation. 
            
            ## Note regarding the conditions ##
            # The input size is [batch_size, n_channels=12, 96, 200]
            # We create a single latent representation for all the conditions together and this single representation is concatenated with the latent varibale z
            # The output encoded shape for the conditions is [batch_size, encoded channels=z_channels default that is set equal to 128, 3, 6]
      
            # Adjust initial convolution to accommodate concatenation
            self.conv_in_concat = nn.Conv2d(
                im_channels + vqvae_encoder.z_channels,  
                self.down_channels[0], 
                kernel_size=3, 
                padding=1
            )
           
        # Scalar Conditioning
        if self.scalar_cond:
            # **Fix the label layer!**
            # Input: `[batch_size,]` (scalar)
            # Output: `[batch_size, t_emb_dim]` (so we can sum with time embedding)
            self.label_layer = nn.Linear(in_features=1, out_features=t_emb_dim)
            
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(Unet_DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                        down_sample=self.down_sample[i], num_layers=self.num_down_layers, attn=self.attns[i]))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(Unet_MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                      num_layers=self.num_mid_layers))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(Unet_UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 64,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
    
    # Define forward method
    def forward(self, x, t, condition):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        
        # **Check if conditions exist**
        if self.image_cond or self.scalar_cond:
            assert condition is not None, "Model requires `condition` when using conditions"
        
        #### 2D Imagine Conditioning ####
        if self.image_cond:
            # Print shape of input condition (before encoding)
            #print(f"Input Condition Shape: {condition.shape}")  # Expecting [batch, 12, 96, 200]
            
            # Encode the condition into the latent space using the encoder (NOT pre-trained)
            im_cond = self.cond_encoder_conv_in(condition)
            for down in self.cond_encoder_layers:
                im_cond = down(im_cond)
            for mid in self.cond_encoder_mids:
                im_cond = mid(im_cond)
            im_cond = self.cond_encoder_norm_out(im_cond)
            im_cond = nn.SiLU()(im_cond)
            im_cond = self.cond_encoder_conv_out(im_cond)
            im_cond = self.cond_encoder_pre_quant(im_cond) # Pre-quantization projection
            im_cond, _, _ = self.cond_encoder_quantize(im_cond)
            
            # Print shape of encoded conditions
            #print(f"Encoded Condition Shape: {im_cond.shape}")  # Expecting [batch, encoded_channels, 3, 6] 
            
            # Ensure correct dimensions
            assert im_cond.shape[-2:] == x.shape[-2:] # Shapes must match
            
            # Print shape of latent Gaussian noise before concatenation
            #print(f"Latent Noise Shape (before concat): {x.shape}")  # Expecting [batch, latent_channels, h, w]
            
            # Concatenate encoded condition with input in latent space
            x = torch.cat([x, im_cond], dim=1)
            
            # Print shape after concatenation
            #print(f"Latent Noise Shape (after concat): {x.shape}")  # Expecting [batch, latent_channels + encoded_channels, h, w]
            
            # B x (C+N) x H x W -> Concatenate the scaled condition and the latent input
            out = self.conv_in_concat(x)
        #### 2D Imagine Conditioning ####
      
        #t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb) # Shape: [batch_size, t_emb_dim]
        
        #### Scalar Conditioning ####
        # Add label embedding to time embedding 
        if self.scalar_cond:
            label_embedding = self.label_layer(condition) 
            t_emb += label_embedding
        #### Scalar Conditioning ####
  
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            #print(out.shape)
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
            
        for mid in self.mids:
            #print(out.shape)
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            #print(out, down_out.shape)
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 64 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
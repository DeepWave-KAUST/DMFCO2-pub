###################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: Linear and Cosine Variance Schedulers  
# 
###################################################################

# Import libraries
import os 
import torch 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import accelerate 
from accelerate import Accelerator

# Instantiate Accelerator
accelerator = Accelerator()

# === Utility function to compute x0 from xt and predicted noise ===
def predict_start_from_noise(x_t, t, noise, alpha_bar):
    sqrt_alpha_bar_t = alpha_bar[t][:, None, None, None].sqrt()
    sqrt_one_minus_alpha_bar_t = (1 - alpha_bar[t])[:, None, None, None].sqrt()
    return (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t

# Define linear variance scheduler class 
class Linear_Variance_Scheduler:
    def __init__(self, time_steps, beta_start, beta_end):
        
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.time_steps).to(accelerator.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
    
    # Define function to plot linear variance schedule
    @staticmethod
    def plot_linear_variance_schedule(time_steps, beta_start=0.0001, beta_end=0.02, figsize=(10, 5), dpi=600, color='blue', linewidth=2.5, save_dir=None):
        scheduler = Linear_Variance_Scheduler(time_steps, beta_start, beta_end)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(scheduler.betas.cpu().numpy(), label="Linear Variance Scheduler", color=color, linewidth=linewidth)
        plt.xlabel("Timesteps", fontsize=10, fontweight='bold')
        plt.ylabel("Beta Value", fontsize=10, fontweight='bold')
        plt.title("Linear Variance Scheduler", fontsize=12, fontweight='bold')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.legend()
        
        # Check if save directory is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, "linear_variance_scheduler.png")
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)  # Save the figure with specified dpi
            plt.close()
    
    # Forward Diffusion Process 
    def diffusion_process(self, x, noise, t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        return predict_start_from_noise(x_t, t, noise, self.alpha_bar)
    
    # Conditional Reverse Diffusion Process -> sampling with a given label 
    def ddpm_sampling(self, model, num_samples, channels, img_size, labels): 
        model.eval()
        with torch.inference_mode():
            #x = torch.randn((num_samples, channels, img_size, img_size)).to(self.device) # Use this line in case height = width 
            x = torch.randn((num_samples, channels, *img_size)).to(accelerator.device) # Using *img_size unpacks the tuple into separate dimensions
            collect = []
            labels = labels.to(accelerator.device)
            for i in tqdm(reversed(range(self.time_steps))):
                t = (torch.ones(num_samples) * i).long().to(accelerator.device)
                pred_noise = model(x, t, labels) 
                alphas = self.alphas[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * pred_noise) +\
                    torch.sqrt(betas) * noise
                if (i+1) % 100 == 0 or i==0:
                    collect.append(x)
        return x, collect

# Define cosine beta schedule
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine variance schedule as proposed in:
    https://openreview.net/forum?id=-NEXDKk8gZ
    
    Args:
        timesteps: Number of diffusion steps (T)
        s: Small offset for numerical stability
        
    Returns:
        betas: Tensor of beta values
    """
    
    import math
    import torch
    
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()

# Define Cosine Variance scheduler class 
class Cosine_Variance_Scheduler:
    def __init__(self, time_steps, s=0.008):
        """
        Cosine variance scheduler for diffusion.
        
        Args:
            time_steps: Number of diffusion steps (T)
            device: Device to store tensors (e.g., 'cuda' or 'cpu')
            s: Offset parameter for cosine beta schedule
        """
        self.time_steps = time_steps
        
        # Use cosine_beta_schedule to calculate betas
        self.betas = cosine_beta_schedule(self.time_steps, s).to(accelerator.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
    
    # Define function to plot cosine variance schedule
    @staticmethod
    def plot_cosine_variance_schedule(time_steps, s=0.008, figsize=(10, 5), dpi=600, color='orange', linewidth=2.5, save_dir=None):
        scheduler = Cosine_Variance_Scheduler(time_steps, s)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(scheduler.betas.cpu().numpy(), label="Cosine Variance Scheduler", color=color, linewidth=linewidth)
        plt.xlabel("Timesteps", fontsize=10, fontweight='bold')
        plt.ylabel("Beta Value", fontsize=10, fontweight='bold')
        plt.title("Cosine Variance Scheduler", fontsize=12, fontweight='bold')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.legend()
        
        # Check if save directory is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, "cosine_variance_scheduler.png")
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)  # Save the figure with specified dpi
            plt.close()
    
    # Forward Diffusion Process 
    def diffusion_process(self, x, noise, t):
        sqrt_alpha_bar = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        return predict_start_from_noise(x_t, t, noise, self.alpha_bar)
    
    # Conditional Reverse Diffusion Process -> sampling with a given label 
    def ddpm_sampling(self, model, num_samples, channels, img_size, labels): 
        model.eval()
        with torch.inference_mode():
            #x = torch.randn((num_samples, channels, img_size, img_size)).to(self.device) # Use this line in case height = width 
            x = torch.randn((num_samples, channels, *img_size)).to(accelerator.device) # Using *img_size unpacks the tuple into separate dimensions
            collect = []
            labels = labels.to(accelerator.device)
            for i in tqdm(reversed(range(self.time_steps))):
                t = (torch.ones(num_samples) * i).long().to(accelerator.device)
                pred_noise = model(x, t, labels) 
                alphas = self.alphas[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alphas) * (x - ((1 - alphas) / (torch.sqrt(1 - alpha_bar))) * pred_noise) +\
                    torch.sqrt(betas) * noise
                if (i+1) % 100 == 0 or i==0:
                    collect.append(x)
        return x, collect
###################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: Package Initialization for dmfco2 folder.
# 
###################################################################

# Dataset Visualization Utilities
from dmfco2.dataset_samples_visualization import (
    dataset_samples_visualization,
    dataset_samples_visualization_64x64, 
    dataset_samples_visualization_no_labels, 
    save_scalar_parameters_to_csv, 
    save_kde_plots_from_csv, 
    create_parameter_gif
)

# Utility Functions
from dmfco2.utils import (
    set_seed,
    dnorm_dP,
    denormalize_co2_train_global,
    denormalize_dP_train_global,
    denormalize_co2_train,
    denormalize_dP_train,
    denormalize_por_train,
    denormalize_co2_val_global,
    denormalize_dP_val_global,
    denormalize_co2_val,
    denormalize_dP_val,
    denormalize_por_val,
    denormalize_co2_test_global,
    denormalize_dP_test_global,
    denormalize_co2_test,
    denormalize_dP_test,
    denormalize_por_test,
    custom_r2_score 
)

# Adversarial Components
from dmfco2.discriminator import Discriminator 
from dmfco2.lpips import LPIPS 

# VQVAE (Vector Quantized VAE)
from dmfco2.vqvae_blocks import (
    VQVAE_DownBlock, 
    VQVAE_MidBlock, 
    VQVAE_UpBlock
)

from dmfco2.vqvae_architecture import VQVAE

from dmfco2.vqvae_training import (
    trainer_vqvae, 
    plot_vqvae_loss_curves, 
    visualize_vqvae_reconstructed_co2_samples_train,
    visualize_vqvae_reconstructed_dP_samples_train,
    visualize_latent_co2_samples_train,
    visualize_latent_dP_samples_train
)

from dmfco2.vqvae_validation import (
    visualize_vqvae_reconstructed_co2_samples_val,
    visualize_vqvae_reconstructed_dP_samples_val,
    visualize_latent_co2_samples_val,
    visualize_latent_dP_samples_val
)

from dmfco2.vqvae_testing import (
    visualize_vqvae_reconstructed_co2_samples_test,
    visualize_vqvae_reconstructed_dP_samples_test,
    visualize_latent_co2_samples_test,
    visualize_latent_dP_samples_test
)

from dmfco2.scheduler import (
    predict_start_from_noise,
    Linear_Variance_Scheduler,
    cosine_beta_schedule,
    Cosine_Variance_Scheduler
)

from dmfco2.time_embedding import get_time_embedding

# U-Net for Diffusion Model
from dmfco2.unet_blocks import (
    Unet_DownBlock, 
    Unet_MidBlock, 
    Unet_UpBlock
)

from dmfco2.unet_architecture import(
    Conditional_Unet
)

# Diffusion Model Training
from dmfco2.diffusion_model_training import (
    trainer_diffusion,
    plot_diffusion_model_loss_curves,
    visualize_gaussian_forward_diffusion_co2_train,
    visualize_gaussian_forward_diffusion_dP_train,
    visualize_reverse_generation_co2_train,
    visualize_reverse_generation_dP_train,
)

from dmfco2.diffusion_model_validation import (
    visualize_reverse_generation_co2_val,
    visualize_reverse_generation_dP_val
)

from dmfco2.diffusion_model_testing import (
    visualize_reverse_generation_co2_test,
    visualize_reverse_generation_dP_test
)

# Package Export List
__all__ = [
    "dataset_samples_visualization",
    "dataset_samples_visualization_64x64",
    "dataset_samples_visualization_no_labels",
    "save_scalar_parameters_to_csv",
    "save_kde_plots_from_csv",
    "create_parameter_gif",
    "set_seed",
    "dnorm_dP",
    "denormalize_co2_train_global",
    "denormalize_dP_train_global",
    "denormalize_co2_train",
    "denormalize_dP_train",
    "denormalize_por_train",
    "denormalize_co2_val_global",
    "denormalize_dP_val_global",
    "denormalize_co2_val",
    "denormalize_dP_val",
    "denormalize_por_val",
    "denormalize_co2_test_global",
    "denormalize_dP_test_global",
    "denormalize_co2_test",
    "denormalize_dP_test",
    "denormalize_por_test",   
    "custom_r2_score",
    "Discriminator",
    "LPIPS",
    "VQVAE_DownBlock",
    "VQVAE_MidBlock",
    "VQVAE_UpBlock",
    "VQVAE",
    "trainer_vqvae",
    "plot_vqvae_loss_curves",
    "visualize_vqvae_reconstructed_co2_samples_train",
    "visualize_vqvae_reconstructed_dP_samples_train",
    "visualize_latent_co2_samples_train",
    "visualize_latent_dP_samples_train",
    "visualize_vqvae_reconstructed_co2_samples_val",
    "visualize_vqvae_reconstructed_dP_samples_val",
    "visualize_latent_co2_samples_val",
    "visualize_latent_dP_samples_val",
    "visualize_vqvae_reconstructed_co2_samples_test",
    "visualize_vqvae_reconstructed_dP_samples_test",
    "visualize_latent_co2_samples_test",
    "visualize_latent_dP_samples_test",
    "predict_start_from_noise",
    "Linear_Variance_Scheduler",
    "cosine_beta_schedule",
    "Cosine_Variance_Scheduler",
    "get_time_embedding",
    "Unet_DownBlock",
    "Unet_MidBlock",
    "Unet_UpBlock",
    "Conditional_Unet",
    "trainer_diffusion",
    "plot_diffusion_model_loss_curves",
    "visualize_gaussian_forward_diffusion_co2_train",
    "visualize_gaussian_forward_diffusion_dP_train",
    "visualize_reverse_generation_co2_train",
    "visualize_reverse_generation_dP_train",
    "visualize_reverse_generation_co2_val",
    "visualize_reverse_generation_dP_val",
    "visualize_reverse_generation_co2_test",
    "visualize_reverse_generation_dP_test",
]
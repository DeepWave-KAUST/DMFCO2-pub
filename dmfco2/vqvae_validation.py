##################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: This script provides functions to visualize the reconstructed samples from a VQ-VAE model for CO₂ gas saturation and pressure build-up validation datasets.
# 
##################################################################

# Import libraries 
import os
import time 
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import accelerate
from accelerate import Accelerator

# Import functions 
from dmfco2.utils import( 
                         denormalize_co2_val, 
                         denormalize_dP_val, 
                         denormalize_por_val,
                         custom_r2_score
                         )

# Define function to visualize latent samples 
def visualize_latent_co2_samples_val(z_tensor, sample_idx, save_dir):
    """
    Visualizes the latent feature map (mean over channels) from VQ-VAE for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both (B, C, H, W) and (C, H, W)
    z_q = z_tensor.detach().cpu().numpy()
    if z_q.ndim == 4:
        z_q = z_q[0]  # Get first sample if batched
    
    # Print the shape of the latent vector
    print(f"Latent shape (C, H, W): {z_q.shape}")

    z_map = z_q.mean(0)  # Average over channels → (H, W)
    
    plt.imshow(z_map, cmap="viridis", interpolation="nearest")
    plt.title(f"Latent Map - CO₂ Gas Saturation (-) @ t = 30.0 y", fontsize=10, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Latent_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight")
    plt.close()
    
# Define function to visualize VQ-VAE reconstructed CO₂ gas saturation samples 
def visualize_vqvae_reconstructed_co2_samples_val(
    model, 
    dataset, 
    sample_indices, 
    min_val_sg, 
    max_val_sg, 
    min_val_por, 
    max_val_por, 
    save_dir
):
    """
    Visualizes reconstructed samples from the VQ-VAE model with metrics and plots for original targets, 
    reconstructed targets, conditions (porosity maps), and differences between original and reconstructed targets.

    Parameters:
    ----------
    model : torch.nn.Module
        Trained VQ-VAE model for reconstruction.
    dataset : torch.utils.data.Dataset
        Dataset containing the original CO₂ gas saturation targets and condition labels.
    sample_indices : list of int
        Indices of the samples to visualize.
    min_train_sg : float
        Minimum value of CO₂ gas saturation in the training dataset.
    max_train_sg : float
        Maximum value of CO₂ gas saturation in the training dataset.
    min_train_por : float
        Minimum value of porosity in the training dataset.
    max_train_por : float
        Maximum value of porosity in the training dataset.
    save_dir : str
        Directory to save the visualization plots.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Set the model to evaluation mode
    model = model.to(accelerator.device)
    model.eval()

    # Convert min and max lists to NumPy arrays
    min_val_sg = np.array(min_val_sg)
    max_val_sg = np.array(max_val_sg)
    min_val_por = min_val_por.cpu().numpy()
    max_val_por = max_val_por.cpu().numpy()
    
    # Define grid for plotting (96x200 resolution)
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))
    
    # Lists to store sample-wise metrics
    mae_list, mse_list, rmse_list, r2_list, mape_list, ssim_list = [], [], [], [], [], []

    # Iterate over the selected indices
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            data, label = dataset[sample_idx]

            data = torch.tensor(data).unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            data = data.to(accelerator.device)
            label = label.to(accelerator.device)

            # Reconstruct using VQ-VAE
            #reconstructed, _, _ = model(data)
            
            # Reconstruct using VQ-VAE (reconstructions+latents)
            reconstructed, z, _ = model(data)

            # Visualize latent space before decoding
            visualize_latent_co2_samples_val(z, sample_idx, save_dir)

            # Fetch the first sample in the batch
            original_target = data[0].cpu().detach().numpy()
            condition_label = label[0].cpu().detach().numpy()
            reconstructed_target = reconstructed[0].cpu().detach().numpy()

            # Denormalize the target and label
            current_min_sg = min_val_sg[sample_idx]
            current_max_sg = max_val_sg[sample_idx]
            denorm_condition = denormalize_por_val(condition_label, min_val_por, max_val_por)
            denorm_original = denormalize_co2_val(original_target, np.array([current_min_sg]), np.array([current_max_sg]))
            denorm_reconstructed = denormalize_co2_val(reconstructed_target, np.array([current_min_sg]), np.array([current_max_sg]))

            # Apply mask on porosity data
            mask = denorm_condition[0, :, :] != 0 
            thickness = np.sum(mask[:, 0]) 

            masked_condition = denorm_condition[0, :, :][mask].reshape((thickness, -1))
            masked_original = denorm_original[0, :, :][mask].reshape((thickness, -1))
            masked_reconstructed = denorm_reconstructed[0, :, :][mask].reshape((thickness, -1))
            masked_difference = np.abs(masked_original - masked_reconstructed)  # Absolute Error 

            # Compute metrics
            threshold = 1e-3  # Ignore values smaller than 1e-4
            valid_mask = masked_original > threshold
            mape = np.mean(np.abs((masked_original[valid_mask] - masked_reconstructed[valid_mask]) / masked_original[valid_mask])) * 100
        
            # Apply nonzero mask for all metrics
            nonzero_mask = masked_original != 0

            # Compute metrics
            mae = mean_absolute_error(masked_original[nonzero_mask], masked_reconstructed[nonzero_mask])
            mse = mean_squared_error(masked_original[nonzero_mask], masked_reconstructed[nonzero_mask])
            rmse = sqrt(mse)
            r2 = custom_r2_score(masked_original, masked_reconstructed)  # Use custom R² for non-zero points. We define ours because the one in scikit-learn is not good.
            ssim_value = ssim(
                masked_original[nonzero_mask],
                masked_reconstructed[nonzero_mask],
                data_range=masked_original[nonzero_mask].max() - masked_original[nonzero_mask].min()
)

            # Store metrics for averaging later
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            mape_list.append(mape)
            ssim_list.append(ssim_value)
            
            # Print metrics
            print(f"Sample {sample_idx + 1}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  SSIM: {ssim_value:.4f}")

            # Define plots with specific vmin/vmax and x-axis limits
            plots = [
                ("Original_Target", masked_original, "Original Sample - CO₂ Gas Saturation (-) @ t = 30.0 y", [0, 3500], (masked_original.min(), masked_original.max())),
                ("Reconstructed_Target", masked_reconstructed, "Reconstructed Sample - CO₂ Gas Saturation (-) @ t = 30.0 y", [0, 3500], (masked_reconstructed.min(), masked_reconstructed.max())),
                ("Absolute_Error", masked_difference, "Absolute Error - CO₂ Gas Saturation (-) @ t = 30.0 y", [0, 3500], (0, 0.25)),
                ("Condition", masked_condition, "Condition - Porosity ($\phi$)", [0, 100000], None),
            ]

            # Generate plots
            for name, masked_data, title, xlim, value_limits in plots:
                vmin, vmax = value_limits if value_limits else (None, None)
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.pcolor(
                    X[:thickness, :], 
                    Y[:thickness, :], 
                    np.flipud(masked_data), 
                    shading="auto", 
                    cmap="jet",
                    vmin=vmin,  
                    vmax=vmax
                )
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
                ax.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
                ax.set_xlim(xlim)  # Dynamically set x-axis limit based on plot type

                # Add metrics text box for the Absolute Error plot
                if name == "Absolute_Error":
                    metrics_text = (
                        f"MAE: {mae:.4f}\n"
                        f"MSE: {mse:.4f}\n"
                        f"RMSE: {rmse:.4f}\n"
                        f"R²: {r2:.4f}\n"
                        f"MAPE: {mape:.2f}%\n"
                        f"SSIM: {ssim_value:.4f}"
                    )
                    ax.text(
                        0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=16, 
                        color="white",  # Set text color to white
                        verticalalignment="bottom", horizontalalignment="right",
                        bbox=dict(boxstyle="round", edgecolor="white")
                        )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"{name}_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight"
                )
                plt.close(fig)
            
            # Generate scatter plot for non-zero values
            true_values = masked_original[nonzero_mask].flatten()
            predicted_values = masked_reconstructed[nonzero_mask].flatten()
            
            # Dynamically determine the range for the 1:1 reference line
            scatter_min = min(true_values.min(), predicted_values.min())  # Minimum of both true and predicted values
            scatter_max = max(true_values.max(), predicted_values.max())  # Maximum of both true and predicted values

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(true_values, predicted_values, alpha=0.8, label="year 30.0", s=15, color='blue')
            ax.plot([scatter_min, scatter_max], [scatter_min, scatter_max], color="red", linestyle="--", label="Regression Line")  # Dynamic regression line
            ax.text(
                0.7, 0.1,
                f"$R^2$: {r2:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8)
            )
            #ax.set_title(f"Scatter Plot", fontsize=12, fontweight="bold")
            ax.set_xlabel("Original CO₂ Gas Saturation (-)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Reconstructed CO₂ Gas Saturation (-)", fontsize=10, fontweight="bold")
            ax.legend()
            ax.grid(linewidth=0.5)
            scatter_plot_path = os.path.join(save_dir, f"Scatter_Sample_{sample_idx + 1}.png")
            plt.tight_layout()
            plt.savefig(scatter_plot_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
    
    # Compute Average Metrics
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)
    avg_mape = np.mean(mape_list)
    avg_ssim = np.mean(ssim_list)

    print("\n====== AVERAGE METRICS ACROSS ALL SAMPLES ======")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save the average metrics in a log file
    with open(os.path.join(save_dir, "vqvae_val_co2_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Reconstruction visualizations saved in '{save_dir}'")
    
# Define function to visualize latent samples 
def visualize_latent_dP_samples_val(z_tensor, sample_idx, save_dir):
    """
    Visualizes the latent feature map (mean over channels) from VQ-VAE for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both (B, C, H, W) and (C, H, W)
    z_q = z_tensor.detach().cpu().numpy()
    if z_q.ndim == 4:
        z_q = z_q[0]  # Get first sample if batched
    
    # Print the shape of the latent vector
    print(f"Latent shape (C, H, W): {z_q.shape}")

    z_map = z_q.mean(0)  # Average over channels → (H, W)
    
    plt.imshow(z_map, cmap="viridis", interpolation="nearest")
    plt.title(f"Latent Map - Pressure Build-Up (-) @ t = 30.0 y", fontsize=10, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Latent_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight")
    plt.close()

# Define function to visualize VQ-VAE reconstructed Pressure Build-Up samples with difference plot
def visualize_vqvae_reconstructed_dP_samples_val(
    model, 
    dataset, 
    sample_indices, 
    min_val_dP, 
    max_val_dP, 
    min_val_por, 
    max_val_por, 
    save_dir
):
    """
    Visualizes reconstructed pressure buildup samples from the VQ-VAE model with separate plots for original 
    targets, reconstructed targets, absolute error, and conditions (porosity maps).

    Parameters:
    ----------
    model : torch.nn.Module
        Trained VQ-VAE model for reconstruction.
    dataset : torch.utils.data.Dataset
        Dataset containing the original pressure buildup targets and condition labels.
    sample_indices : list of int
        Indices of the samples to visualize.
    min_train_dP : list or array-like
        List or array of minimum values for pressure buildup in the training dataset (used for denormalization).
    max_train_dP : list or array-like
        List or array of maximum values for pressure buildup in the training dataset (used for denormalization).
    min_train_por : torch.Tensor or array-like
        Minimum value for porosity in the training dataset (used for denormalization).
    max_train_por : torch.Tensor or array-like
        Maximum value for porosity in the training dataset (used for denormalization).
    save_dir : str
        Directory to save the visualization plots.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Set the model to evaluation mode
    model = model.to(accelerator.device)
    model.eval()

    # Convert min and max lists to NumPy arrays
    min_val_dP = np.array(min_val_dP)
    max_val_dP = np.array(max_val_dP)
    min_val_por = min_val_por.cpu().numpy()
    max_val_por = max_val_por.cpu().numpy()

    # Define grid for plotting (96x200 resolution)
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))
    
    # Lists to store sample-wise metrics
    mae_list, mse_list, rmse_list, r2_list, mape_list, ssim_list = [], [], [], [], [], []

    # Iterate over the selected indices
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            data, label = dataset[sample_idx]
            
            # Use the accelerator to place data and label on the correct device
            data = torch.tensor(data).unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            data = data.to(accelerator.device)
            label = label.to(accelerator.device)

            # Reconstruct using VQ-VAE
            #reconstructed, _, _ = model(data)
            
            # Reconstruct using VQ-VAE (reconstructions+latents)
            reconstructed, z, _ = model(data)

            # Visualize latent space before decoding
            visualize_latent_dP_samples_val(z, sample_idx, save_dir)

            # Fetch the first sample in the batch
            original_target = data[0].cpu().detach().numpy()
            condition_label = label[0].cpu().detach().numpy()
            reconstructed_target = reconstructed[0].cpu().detach().numpy()

            # Denormalize the target and label
            current_min_dP = min_val_dP[sample_idx]
            current_max_dP = max_val_dP[sample_idx]
            denorm_condition = denormalize_por_val(condition_label, min_val_por, max_val_por)
            denorm_original = denormalize_dP_val(original_target, np.array([current_min_dP]), np.array([current_max_dP]))
            denorm_reconstructed = denormalize_dP_val(reconstructed_target, np.array([current_min_dP]), np.array([current_max_dP]))

            # Apply mask on porosity data
            mask = denorm_condition[0, :, :] != 0 
            thickness = np.sum(mask[:, 0]) 

            masked_condition = denorm_condition[0, :, :][mask].reshape((thickness, -1))
            masked_original = denorm_original[0, :, :][mask].reshape((thickness, -1))
            masked_reconstructed = denorm_reconstructed[0, :, :][mask].reshape((thickness, -1))
            
            # Compute absolute error
            masked_error = np.abs(masked_original - masked_reconstructed)  
            
            # Compute metrics
            threshold = 1e-2  # Ignore values smaller than 1e-2
            valid_mask = masked_original > threshold
            mape = np.mean(np.abs((masked_original[valid_mask] - masked_reconstructed[valid_mask]) / masked_original[valid_mask])) * 100
            
            # Apply nonzero mask for all metrics
            nonzero_mask = masked_original != 0

            # Compute metrics
            mae = mean_absolute_error(masked_original[nonzero_mask], masked_reconstructed[nonzero_mask])
            mse = mean_squared_error(masked_original[nonzero_mask], masked_reconstructed[nonzero_mask])
            rmse = sqrt(mse)
            r2 = custom_r2_score(masked_original, masked_reconstructed)  # Use custom R² for non-zero points. We define ours because the one in scikit-learn is not good.
            ssim_value = ssim(
                masked_original[nonzero_mask],
                masked_reconstructed[nonzero_mask],
                data_range=masked_original[nonzero_mask].max() - masked_original[nonzero_mask].min())
            
            # Store metrics for averaging later
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            mape_list.append(mape)
            ssim_list.append(ssim_value)
            
            # Print metrics
            print(f"Sample {sample_idx + 1}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  SSIM: {ssim_value:.4f}")
            
            # Define plots with specific ranges
            plots = [
                ("Original_Target", masked_original, "Original Sample - Pressure Build-Up (bar) @ t = 30.0 y", [0, 3500], (masked_original.min(), masked_original.max())),
                ("Reconstructed_Target", masked_reconstructed, "Reconstructed Sample - Pressure Build-Up (bar) @ t = 30.0 y", [0, 3500], (masked_reconstructed.min(), masked_reconstructed.max())),
                ("Absolute_Error", masked_error, "Absolute Error - Pressure Build-Up (bar) @ t = 30.0 y", [0, 3500], None),
                ("Condition", masked_condition, "Condition - Porosity ($\phi$)", [0, 100000], None),
            ]

            # Generate plots
            for name, masked_data, title, xlim, value_limits in plots:
                vmin, vmax = value_limits if value_limits else (None, None)
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.pcolor(
                    X[:thickness, :], 
                    Y[:thickness, :], 
                    np.flipud(masked_data), 
                    shading="auto", 
                    cmap="jet",
                    vmin=vmin,  
                    vmax=vmax
                )
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
                ax.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
                ax.set_xlim(xlim)  # Dynamically set x-axis limit based on plot type
                
                # Add metrics text box for the Absolute Error plot
                if name == "Absolute_Error":
                    metrics_text = (
                        f"MAE: {mae:.4f}\n"
                        f"MSE: {mse:.4f}\n"
                        f"RMSE: {rmse:.4f}\n"
                        f"R²: {r2:.4f}\n"
                        f"MAPE: {mape:.2f}%\n"
                        f"SSIM: {ssim_value:.4f}"
                    )
                    ax.text(
                        0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=16, 
                        color="white",  # Set text color to white
                        verticalalignment="bottom", horizontalalignment="right",
                        bbox=dict(boxstyle="round", edgecolor="white")
                        )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"{name}_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight"
                )
                plt.close(fig)
                
            # Generate scatter plot for non-zero values
            true_values = masked_original[nonzero_mask].flatten()
            predicted_values = masked_reconstructed[nonzero_mask].flatten()
            
            # Dynamically determine the range for the 1:1 reference line
            scatter_min = min(true_values.min(), predicted_values.min())  # Minimum of both true and predicted values
            scatter_max = max(true_values.max(), predicted_values.max())  # Maximum of both true and predicted values

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(true_values, predicted_values, alpha=0.8, label="year 30.0", s=15, color='blue')
            ax.plot([scatter_min, scatter_max], [scatter_min, scatter_max], color="red", linestyle="--", label="Regression Line")  # Dynamic regression line
            ax.text(
                0.7, 0.1,
                f"$R^2$: {r2:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8)
            )
            #ax.set_title(f"Scatter Plot", fontsize=12, fontweight="bold")
            ax.set_xlabel("Original Pressure Build-Up (bar)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Reconstructed Pressure Build-Up (bar)", fontsize=10, fontweight="bold")
            ax.legend()
            ax.grid(linewidth=0.5)
            scatter_plot_path = os.path.join(save_dir, f"Scatter_Sample_{sample_idx + 1}.png")
            plt.tight_layout()
            plt.savefig(scatter_plot_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
    
    # Compute Average Metrics
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)
    avg_mape = np.mean(mape_list)
    avg_ssim = np.mean(ssim_list)

    print("\n====== AVERAGE METRICS ACROSS ALL SAMPLES ======")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save the average metrics in a log file
    with open(os.path.join(save_dir, "vqvae_val_dP_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Reconstruction visualizations saved in '{save_dir}'")
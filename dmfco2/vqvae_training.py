##################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: This script provides functions to visualize the reconstructed samples from a VQ-VAE model for CO₂ gas saturation and pressure build-up training datasets.
# 
##################################################################

# Import libraries 
import os
import time 
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import accelerate
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

# Import functions 
from dmfco2.utils import( 
                         denormalize_co2_train, 
                         denormalize_dP_train, 
                         denormalize_por_train,
                         custom_r2_score)

# Trainer function for VQVAE
def trainer_vqvae(model, 
                  lpips_model, 
                  discriminator,
                  recon_criterion, 
                  disc_criterion,
                  optimizer_g, 
                  optimizer_d,
                  training_dataloader, 
                  num_epochs, 
                  disc_step_start, 
                  save_dir, 
                  save_dir_results,
                  summary_file_path,
                  trial_params=None,
                  save_loss_curves=False,
                  loss_weights=None,
                  early_stopping_patience=10):
    
    # Define accelerator
    accelerator = Accelerator(split_batches=True) 
    
    # Prepare models, optimizers, and dataloaders
    model, optimizer_g, optimizer_d, training_dataloader = accelerator.prepare(model, optimizer_g, optimizer_d, training_dataloader)
    lpips_model = lpips_model.to(accelerator.device)
    discriminator = accelerator.prepare(discriminator)
    
    # Initialize TensorBoard writer if save_loss_curves is True (Full training)
    if save_loss_curves:
        log_dir = os.path.join(save_dir_results, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        accelerator.print(f"📌 TensorBoard Logging Enabled at: {log_dir}")
    
    # Create generator and discriminator loss
    g_loss_list, disc_loss_list = [], []
    step_count = 0
    
    # Start the timer
    start_time = time.time()
    
    # Early Stopping Tracking 
    best_g_loss = float('inf')
    epochs_since_improvement = 0
    
    # Open the summary file
    with open(summary_file_path, 'w') as summary_file:
        
        # Write trial parameters at the top of the summary file
        if trial_params:
            summary_file.write("Trial Parameters:\n")
            for param, value in trial_params.items():
                summary_file.write(f"{param}: {value}\n")
            summary_file.write("\n")  # Add a blank line for readability
        
        # Write header
        summary_file.write("Epoch\tGenerator Loss\tDiscriminator Loss\n")
        
        for epoch_idx in range(num_epochs):
            g_losses, disc_losses = [], []
            
            for data, _ in tqdm(training_dataloader):
                model.train()
                discriminator.train()
                
                # GPU monitoring
                #accelerator.print(f"Using device: {accelerator.device} | Process {accelerator.process_index}")
                
                step_count += 1
                data = data.to(accelerator.device)
        
                ####### Generator (VQVAE) #######
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                
                # Forward Pass through VQVAE
                output, z, quantize_losses = model(data)
                
                # Reshape output to match original shape (96, 200) - Here when we downsample we do not obtain perfect values 
                #output = F.interpolate(output, size=(96, 200), mode="bilinear", align_corners=True)
                
                # Reconstruction Loss
                recon_loss = recon_criterion(output, data)
                
                vqvae_loss = (recon_loss * loss_weights["recon_loss"]  +
                              (loss_weights["codebook_loss"] * quantize_losses['codebook_loss']) +
                              (loss_weights["commitment_loss"] * quantize_losses['commitment_loss']))
                
                # LPIPS Loss
                lpips_loss = torch.mean(lpips_model(output, data))
                
                # Total Generator Loss
                g_loss = vqvae_loss + (loss_weights["lpips_loss"] * lpips_loss)
                
                # Adversarial Loss
                if step_count > disc_step_start:
                    disc_fake_pred = discriminator(output)
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.ones_like(disc_fake_pred))
                    g_loss += loss_weights["disc_fake_loss"] * disc_fake_loss
                
                # Backward pass with Accelerator
                accelerator.backward(g_loss)
                optimizer_g.step()
                
                # Append Generator Loss
                g_losses.append(g_loss.item())
                #####################################

                ######### Optimize Discriminator #######
                if step_count > disc_step_start:
                    fake = output.detach()
                    disc_fake_pred = discriminator(fake)
                    disc_real_pred = discriminator(data)
                    
                    #print(f"disc_fake_pred: {disc_fake_pred.mean().item()}, disc_real_pred: {disc_real_pred.mean().item()}")
                    
                    disc_fake_loss = disc_criterion(disc_fake_pred,
                                                    torch.zeros_like(disc_fake_pred))
                    disc_real_loss = disc_criterion(disc_real_pred,
                                                    torch.ones_like(disc_real_pred))
                    
                    #print(f"disc_fake_loss: {disc_fake_loss.item()}, disc_real_loss: {disc_real_loss.item()}")
                    
                    disc_loss = (disc_fake_loss + disc_real_loss) * loss_weights["disc_loss"]
                    accelerator.backward(disc_loss)
                    optimizer_d.step()
                    disc_losses.append(disc_loss.item())
                #####################################

            # Per-epoch Loss Averages
            g_epoch_loss = torch.tensor(g_losses).mean().item()
            disc_epoch_loss = torch.tensor(disc_losses).mean().item() if disc_losses else 0.0
            
            g_loss_list.append(g_epoch_loss)
            disc_loss_list.append(disc_epoch_loss)
            
            print(f"Epoch: {epoch_idx+1}/{num_epochs} | Generator Loss: {g_epoch_loss:.4f} | Discriminator Loss: {disc_epoch_loss:.4f}")
            
            # Write epoch results to the summary file
            summary_file.write(f"{epoch_idx+1}\t{g_epoch_loss:.4f}\t{disc_epoch_loss:.4f}\n")
            
            # Log losses to TensorBoard
            if save_loss_curves:
                writer.add_scalars(
                    "VQ-VAE/Losses",  # Single tag for both losses
                    {"Generator Loss": g_epoch_loss, "Discriminator Loss": disc_epoch_loss},
                    epoch_idx,
                )
            
            # Early stopping check
            if g_epoch_loss < best_g_loss:
                best_g_loss = g_epoch_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch_idx+1} (no improvement for {early_stopping_patience} epochs).")
                    summary_file.write(f"\nTraining stopped early at epoch {epoch_idx+1} due to no improvement for {early_stopping_patience} epochs.\n")
                    break
        
        # End the timer
        end_time = time.time()
        total_time = end_time - start_time
        
        # Write total training time to the summary file
        summary_file.write("\nTotal Training Time:\n")
        summary_file.write(f"{total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s\n")
        
        # Print the total training time
        print(f"\nTotal Training Time: {total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s")
    
    # Plot loss curves if enabled, after all epochs are completed
    if save_loss_curves:
        plot_vqvae_loss_curves(g_loss_list, disc_loss_list, save_dir_results, filename="vqvae_loss_curves.png")
        # Close TensorBoard writer if it exists
        if 'writer' in locals():
            writer.close()
    
    return g_loss_list, disc_loss_list

# Define function to plot and save VQ-VAE loss curves
def plot_vqvae_loss_curves(g_loss_list, disc_loss_list, save_dir, filename="vqvae_loss_curves.png"):
    """
    Plots and saves the loss curves for the VQ-VAE training process, including generator and discriminator losses.

    Parameters:
    ----------
    vqvae_loss_list : list or np.array
        List of VQVAE losses per epoch.
    lpips_loss_list : list or np.array
        List of LPIPS losses per epoch.
    disc_loss_list : list or np.array
        List of discriminator losses per epoch.
    save_dir : str
        Directory to save the loss curve plot.
    filename : str, optional
        Filename for the saved plot. Default is "vqvae_loss_curves.png".

    Returns:
    -------
    None
        Saves the loss curve plot as a high-resolution PNG image in the specified directory.

    Notes:
    -----
    - The function combines VQVAE and LPIPS losses to compute the total generator loss.
    - The plot includes separate curves for generator and discriminator losses.
    - The loss curve is saved in the specified directory as a PNG file.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert loss lists to numpy arrays
    g_loss_list = np.array(g_loss_list)
    disc_loss_list = np.array(disc_loss_list)

    # Define epochs
    epochs = np.arange(1, len(g_loss_list) + 1)

    # Create plot
    plt.figure(figsize=(12, 8))  # Increase the figure size for better readability

    # Plot each loss curve
    plt.plot(epochs, g_loss_list, '-', color='red', linewidth=2, label='Generator Loss')
    plt.plot(epochs, disc_loss_list, '-', color='blue', linewidth=2, label='Discriminator Loss')

    # Add title and labels with enhanced font sizes and styles
    plt.title('Loss Curves Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')

    # Enable grid with light color and dashed style for readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Set tick parameters and limit x-axis to whole epochs
    plt.xticks(np.arange(1, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    
    # Use semi-logarithmic scaling for y-axis
    plt.semilogy()

    # Add legend with a larger font size and light background
    plt.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.8, facecolor='white')

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)  # Save the figure with specified dpi
    plt.close()

    print(f"Loss curve saved at: {save_path}")

# Define function to visualize latent samples 
def visualize_latent_co2_samples_train(z_tensor, sample_idx, save_dir):
    """
    Visualizes the latent feature map (mean over channels) from VQ-VAE for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both (B, C, H, W) and (C, H, W)
    z_q = z_tensor.detach().cpu().numpy()
    if z_q.ndim == 4:
        z_q = z_q[0]  # Get first sample if batched
    
    # Print the shape of the latent vector
    #print(f"Latent shape (C, H, W): {z_q.shape}")

    z_map = z_q.mean(0)  # Average over channels → (H, W)
    
    plt.imshow(z_map, cmap="viridis", interpolation="nearest")
    plt.title(f"Latent Map - CO₂ Gas Saturation (-) @ t = 30.0 y", fontsize=10, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Latent_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight")
    plt.close()

# Define function to visualize VQ-VAE reconstructed CO₂ gas saturation samples 
def visualize_vqvae_reconstructed_co2_samples_train(
    model, 
    dataset, 
    sample_indices, 
    min_train_sg, 
    max_train_sg, 
    min_train_por, 
    max_train_por, 
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
    min_train_sg = np.array(min_train_sg)
    max_train_sg = np.array(max_train_sg)
    min_train_por = min_train_por.cpu().numpy()
    max_train_por = max_train_por.cpu().numpy()
    
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

            # Reconstruct using VQ-VAE (only reconstructions)
            #reconstructed, _, _ = model(data)
            
            # Reconstruct using VQ-VAE (reconstructions+latents)
            reconstructed, z, _ = model(data)

            # Visualize latent space before decoding
            visualize_latent_co2_samples_train(z, sample_idx, save_dir)

            # Debugging: Print the output shape
            #print(f"Sample {sample_idx} - Raw Model Output Shape:", reconstructed.shape)

            # Fetch the first sample in the batch
            original_target = data[0].cpu().detach().numpy()
            condition_label = label[0].cpu().detach().numpy()
            reconstructed_target = reconstructed[0].cpu().detach().numpy()

            # Denormalize the target and label
            current_min_sg = min_train_sg[sample_idx]
            current_max_sg = max_train_sg[sample_idx]
            denorm_condition = denormalize_por_train(condition_label, min_train_por, max_train_por)
            denorm_original = denormalize_co2_train(original_target, np.array([current_min_sg]), np.array([current_max_sg]))
            denorm_reconstructed = denormalize_co2_train(reconstructed_target, np.array([current_min_sg]), np.array([current_max_sg]))

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
    with open(os.path.join(save_dir, "vqvae_train_co2_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Reconstruction visualizations saved in '{save_dir}'")

# Define function to visualize latent samples 
def visualize_latent_dP_samples_train(z_tensor, sample_idx, save_dir):
    """
    Visualizes the latent feature map (mean over channels) from VQ-VAE for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both (B, C, H, W) and (C, H, W)
    z_q = z_tensor.detach().cpu().numpy()
    if z_q.ndim == 4:
        z_q = z_q[0]  # Get first sample if batched
    
    # Print the shape of the latent vector
    #print(f"Latent shape (C, H, W): {z_q.shape}")

    z_map = z_q.mean(0)  # Average over channels → (H, W)
    
    plt.imshow(z_map, cmap="viridis", interpolation="nearest")
    plt.title(f"Latent Map - Pressure Build-Up (-) @ t = 30.0 y", fontsize=10, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Latent_Sample_{sample_idx + 1}.png"), dpi=600, bbox_inches="tight")
    plt.close()
    
# Define function to visualize VQ-VAE reconstructed Pressure Build-Up samples 
def visualize_vqvae_reconstructed_dP_samples_train(
    model, 
    dataset, 
    sample_indices, 
    min_train_dP, 
    max_train_dP, 
    min_train_por, 
    max_train_por, 
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
    min_train_dP = np.array(min_train_dP)
    max_train_dP = np.array(max_train_dP)
    min_train_por = min_train_por.cpu().numpy()
    max_train_por = max_train_por.cpu().numpy()

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
            visualize_latent_dP_samples_train(z, sample_idx, save_dir)
            
            # Debugging: Print the output shape
            print(f"Sample {sample_idx} - Raw Model Output Shape:", reconstructed.shape)

            # Fetch the first sample in the batch
            original_target = data[0].cpu().detach().numpy()
            condition_label = label[0].cpu().detach().numpy()
            reconstructed_target = reconstructed[0].cpu().detach().numpy()

            # Denormalize the target and label
            current_min_dP = min_train_dP[sample_idx]
            current_max_dP = max_train_dP[sample_idx]
            denorm_condition = denormalize_por_train(condition_label, min_train_por, max_train_por)
            denorm_original = denormalize_dP_train(original_target, np.array([current_min_dP]), np.array([current_max_dP]))
            denorm_reconstructed = denormalize_dP_train(reconstructed_target, np.array([current_min_dP]), np.array([current_max_dP]))

            # Apply mask on porosity data
            mask = denorm_condition[0, :, :] != 0 
            thickness = np.sum(mask[:, 0]) 
            
            # Debugging: Print shapes
            #print("Shape of denorm_condition:", denorm_condition[0, :, :].shape)
            #print("Shape of denorm_original:", denorm_original[0, :, :].shape)
            #print("Shape of denorm_reconstructed:", denorm_reconstructed[0, :, :].shape)
            #print("Shape of mask:", mask.shape)

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
    with open(os.path.join(save_dir, "vqvae_train_dP_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")

    print(f"Reconstruction visualizations saved in '{save_dir}'")
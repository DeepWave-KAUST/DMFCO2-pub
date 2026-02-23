###################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: This script implements training and visualization functions for the diffusion model applied to CO₂ gas saturation and pressure build-up prediction.
# 
###################################################################

# Import libraries 
import os
import time 
import torch
import time
import numpy as np 
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import accelerate
from accelerate import Accelerator

from torch.utils.tensorboard import SummaryWriter

# Import functions
from dmfco2.utils import(
    denormalize_co2_train, 
    denormalize_por_train, 
    denormalize_dP_train,
    custom_r2_score
)

# Trainer function for diffusion model (latent loss + pixel loss)
def trainer_diffusion(model_unet, 
                      vqvae_model, 
                      ddpm, 
                      criterion, 
                      optimizer, 
                      training_dataloader, 
                      validation_dataloader, 
                      testing_dataloader, 
                      n_epochs, 
                      save_dir, 
                      save_dir_results,
                      summary_file_path, 
                      trial_params=None,
                      save_loss_curves=False,
                      lambda_pixel=1.0,
                      early_stopping_patience=10):
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # VQVAE model in eval mode
    vqvae_model.eval().to(accelerator.device)
    
    # Prepare with Accelerator
    model_unet, optimizer, training_dataloader, validation_dataloader, testing_dataloader = accelerator.prepare(
        model_unet, 
        optimizer, 
        training_dataloader, 
        validation_dataloader, 
        testing_dataloader
    )
    
    # Initialize TensorBoard writer if save_loss_curves is True
    if save_loss_curves:
        log_dir = os.path.join(save_dir_results, "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        accelerator.print(f"📌 TensorBoard Logging Enabled at: {log_dir}")

    training_loss, validation_loss, test_loss = [], [], []
    
    # Start the timer
    start_time = time.time()
    
    # Early stopping check
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    
    # Open the summary file
    with open(summary_file_path, 'w') as summary_file:
        if trial_params:
            summary_file.write("Trial Parameters:\n")
            for param, value in trial_params.items():
                summary_file.write(f"{param}: {value}\n")
            summary_file.write("\n")

        summary_file.write("Epoch\tTraining Loss\tValidation Loss\tTest Loss\n")

        for epoch in range(n_epochs):
            training_losses, validation_losses, test_losses = [], [], []

            # === Training ===
            for data, labels in tqdm(training_dataloader):
                model_unet.train()
                data, labels = data.to(accelerator.device), labels.to(accelerator.device)

                with torch.no_grad():
                    latent, _ = vqvae_model.encode(data)

                t = torch.randint(low=0, high=1000, size=(latent.shape[0],)).to(accelerator.device)
                noise = torch.randn_like(latent)
                xt = ddpm.diffusion_process(x=latent, noise=noise, t=t)
                
                # === Latent loss ===
                pred_noise = model_unet(xt, t, labels)
                loss_latent = criterion(noise, pred_noise)

                # === Pixel-space loss ===
                with torch.no_grad():
                    sqrt_alpha_bar = ddpm.sqrt_alpha_bar[t][:, None, None, None]
                    sqrt_one_minus_alpha_bar = ddpm.sqrt_one_minus_alpha_bar[t][:, None, None, None]

                    noisy_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * noise
                    reconstructed_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * pred_noise

                    decoded_xt = vqvae_model.decode(noisy_latent)
                    decoded_z0 = vqvae_model.decode(reconstructed_latent)

                loss_pixel = criterion(decoded_xt, decoded_z0)
                total_loss = loss_latent + lambda_pixel * loss_pixel
                
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                optimizer.step()
                training_losses.append(total_loss.item())

            training_loss.append(np.mean(training_losses))

            # === Validation ===
            with torch.no_grad():
                for data, labels in tqdm(validation_dataloader):
                    model_unet.eval()
                    data, labels = data.to(accelerator.device), labels.to(accelerator.device)
                    latent, _ = vqvae_model.encode(data)

                    t = torch.randint(low=0, high=1000, size=(latent.shape[0],)).to(accelerator.device)
                    noise = torch.randn_like(latent)
                    xt = ddpm.diffusion_process(x=latent, noise=noise, t=t)

                    pred_noise = model_unet(xt, t, labels)
                    loss_latent = criterion(noise, pred_noise)

                    sqrt_alpha_bar = ddpm.sqrt_alpha_bar[t][:, None, None, None]
                    sqrt_one_minus_alpha_bar = ddpm.sqrt_one_minus_alpha_bar[t][:, None, None, None]
                    noisy_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * noise
                    reconstructed_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * pred_noise

                    decoded_xt = vqvae_model.decode(noisy_latent)
                    decoded_z0 = vqvae_model.decode(reconstructed_latent)

                    loss_pixel = criterion(decoded_xt, decoded_z0)
                    total_loss = loss_latent + lambda_pixel * loss_pixel
                    validation_losses.append(total_loss.item())

            validation_loss.append(np.mean(validation_losses))
            
            # === Early Stopping Logic ===
            current_val_loss = validation_loss[-1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs).")
                    break

            # === Testing ===
            with torch.no_grad():
                for data, labels in tqdm(testing_dataloader):
                    model_unet.eval()
                    data, labels = data.to(accelerator.device), labels.to(accelerator.device)
                    latent, _ = vqvae_model.encode(data)

                    t = torch.randint(low=0, high=1000, size=(latent.shape[0],)).to(accelerator.device)
                    noise = torch.randn_like(latent)
                    xt = ddpm.diffusion_process(x=latent, noise=noise, t=t)

                    pred_noise = model_unet(xt, t, labels)
                    loss_latent = criterion(noise, pred_noise)

                    sqrt_alpha_bar = ddpm.sqrt_alpha_bar[t][:, None, None, None]
                    sqrt_one_minus_alpha_bar = ddpm.sqrt_one_minus_alpha_bar[t][:, None, None, None]
                    noisy_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * noise
                    reconstructed_latent = sqrt_alpha_bar * latent + sqrt_one_minus_alpha_bar * pred_noise

                    decoded_xt = vqvae_model.decode(noisy_latent)
                    decoded_z0 = vqvae_model.decode(reconstructed_latent)

                    loss_pixel = criterion(decoded_xt, decoded_z0)
                    total_loss = loss_latent + lambda_pixel * loss_pixel
                    test_losses.append(total_loss.item())
                    
            test_loss.append(np.mean(test_losses))

            # === Logging ===
            print(f"Epoch: {epoch+1}/{n_epochs}\t| Training loss: {training_loss[-1]:.4f} | Validation loss: {validation_loss[-1]:.4f} | Test loss: {test_loss[-1]:.4f}")
            summary_file.write(f"{epoch+1}\t{training_loss[-1]:.4f}\t{validation_loss[-1]:.4f}\t{test_loss[-1]:.4f}\n")

            if save_loss_curves:
                writer.add_scalars("Diffusion/Losses", {
                    "Training Loss": training_loss[-1],
                    "Validation Loss": validation_loss[-1],
                    "Test Loss": test_loss[-1],
                }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        summary_file.write("\nTotal Training Time:\n")
        summary_file.write(f"{total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s\n")
        print(f"\nTotal Training Time: {total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s")

    if save_loss_curves:
        plot_diffusion_model_loss_curves(training_loss, validation_loss, test_loss, save_dir_results, filename="diffusion_model_loss_curves.png")
        if 'writer' in locals():
            writer.close()

    return training_loss, validation_loss, test_loss

# Function to plot and save diffusion model loss curves
def plot_diffusion_model_loss_curves(training_loss, validation_loss, test_loss, save_dir, filename="diffusion_model_loss_curves.png"):
    """
    Plots and saves the loss curves for the diffusion model training process.

    Parameters:
    ----------
    training_loss : list or np.array
        List of training losses per epoch.
    validation_loss : list or np.array
        List of validation losses per epoch.
    test_loss : list or np.array
        List of test losses per epoch.
    save_dir : str
        Directory to save the loss curve plot.
    filename : str, optional
        Filename for the saved plot. Default is "diffusion_model_loss_curves.png".

    Returns:
    -------
    None
        Saves the loss curve plot as a high-resolution PNG image in the specified directory.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define epochs
    n_epochs = len(training_loss)
    epochs = np.arange(1, n_epochs + 1)

    # Create plot
    plt.figure(figsize=(10, 6))  # Increase the figure size for better readability

    # Plot each loss curve
    plt.plot(epochs, training_loss, '-', color='magenta', linewidth=2, label='Training Loss')
    plt.plot(epochs, validation_loss, '-', color='lime', linewidth=2, label='Validation Loss')
    plt.plot(epochs, test_loss, '-', color='blue', linewidth=2, label='Test Loss')

    # Add title and labels with enhanced font sizes and styles
    plt.title('MSE Loss Curve Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10, fontweight='bold')
    plt.ylabel('Loss', fontsize=10, fontweight='bold')

    # Enable grid with light color and dashed style for readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Set tick parameters and limit x-axis to whole epochs
    plt.xticks(np.arange(1, n_epochs + 1, step=max(1, n_epochs // 10)))

    # Use semi-logarithmic scaling for y-axis
    plt.semilogy()

    # Add legend with a larger font size and light background
    plt.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.8, facecolor='white')

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)  # Save the figure with specified dpi
    plt.close()

    print(f"Diffusion model loss curve saved at: {save_path}")

# Define function to visualize gaussian forward diffusion 
def visualize_gaussian_forward_diffusion_co2_train(
    ddpm,  # Diffusion scheduler
    dataset,  # Dataset for dataset to visualize
    sample_indices,  # Indices of samples to visualize within the batch
    t_steps,  # List of timesteps for forward diffusion
    save_dir,  # Directory to save visualizations
    vqvae_model,  # Path to VQ-VAE checkpoint 
):
    """
    Visualize forward diffusion in latent space with masking applied for CO₂ gas saturation, 
    and separately plot the target and condition maps for a specific batch.

    Parameters:
    ----------
    - See above for details.

    Returns:
    -------
    None
        Saves visualizations in the specified directory.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Load Pretrained VQVAE
    vqvae_model.eval().to(accelerator.device)  # Set to evaluation mode

    # Fetch specific samples from dataset
    targets, labels = [], []
    for idx in sample_indices:
        target, label = dataset[idx]
        targets.append(target)
        labels.append(label)
    
    # Convert to tensors and move to device
    targets = torch.stack(targets).to(accelerator.device)
    labels = torch.stack(labels).to(accelerator.device)

    # Iterate over samples in the selected batch
    for i, (target, label) in enumerate(zip(targets, labels)):
        dataset_index = sample_indices[i]  # Restore this line
        
        # Encode the sample into latent space
        with torch.no_grad():
            latent, _ = vqvae_model.encode(target.unsqueeze(0))
        
        # Apply forward diffusion individually per timestep
        noisy_latent_list = []
        for t in t_steps:
            noise = torch.randn_like(latent)
            t_tensor = torch.tensor([t], device=latent.device)
            noisy = ddpm.diffusion_process(x=latent, noise=noise, t=t_tensor)
            noisy_latent_list.append(noisy)

        # Stack the noisy latents for visualization
        noisy_latent_stack = torch.cat(noisy_latent_list, dim=0)  # shape: (T, C, H, W)

        # Visualize the noisy latent tensors directly (mean over channels)
        fig, axes = plt.subplots(len(t_steps), 1, figsize=(10, 4 * len(t_steps)))

        for idx, noisy_latent in enumerate(noisy_latent_stack):
            latent_map = noisy_latent.mean(0).cpu().numpy()  # [H, W]
            ax = axes[idx]
            im = ax.imshow(latent_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f"Latent Forward Diffusion - Time Step {t_steps[idx]}", fontsize=12, fontweight="bold")
            ax.axis("off")

        # Save the latent progression plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Forward_Diffusion_Step.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
        
    print(f"Visualizations saved in '{save_dir}'")

# Define function to visualize gaussian forward diffusion 
def visualize_gaussian_forward_diffusion_dP_train(
    ddpm,  # Diffusion scheduler
    dataset,  # Dataset for dataset to visualize
    sample_indices,  # Indices of samples to visualize within the batch
    t_steps,  # List of timesteps for forward diffusion
    save_dir,  # Directory to save visualizations
    vqvae_model,  # Path to VQ-VAE checkpoint 
):
    """
    Visualize forward diffusion in latent space with masking applied for CO₂ gas saturation, 
    and separately plot the target and condition maps for a specific batch.

    Parameters:
    ----------
    - See above for details.

    Returns:
    -------
    None
        Saves visualizations in the specified directory.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Load Pretrained VQVAE
    vqvae_model.eval().to(accelerator.device)  # Set to evaluation mode

    # Fetch specific samples from dataset
    targets, labels = [], []
    for idx in sample_indices:
        target, label = dataset[idx]
        targets.append(target)
        labels.append(label)
    
    # Convert to tensors and move to device
    targets = torch.stack(targets).to(accelerator.device)
    labels = torch.stack(labels).to(accelerator.device)

    # Iterate over samples in the selected batch
    for i, (target, label) in enumerate(zip(targets, labels)):
        dataset_index = sample_indices[i]  # Restore this line
        
        # Encode the sample into latent space
        with torch.no_grad():
            latent, _ = vqvae_model.encode(target.unsqueeze(0))
        
        # Apply forward diffusion individually per timestep
        noisy_latent_list = []
        for t in t_steps:
            noise = torch.randn_like(latent)
            t_tensor = torch.tensor([t], device=latent.device)
            noisy = ddpm.diffusion_process(x=latent, noise=noise, t=t_tensor)
            noisy_latent_list.append(noisy)

        # Stack the noisy latents for visualization
        noisy_latent_stack = torch.cat(noisy_latent_list, dim=0)  # shape: (T, C, H, W)

        # Visualize the noisy latent tensors directly (mean over channels)
        fig, axes = plt.subplots(len(t_steps), 1, figsize=(10, 4 * len(t_steps)))

        for idx, noisy_latent in enumerate(noisy_latent_stack):
            latent_map = noisy_latent.mean(0).cpu().numpy()  # [H, W]
            ax = axes[idx]
            im = ax.imshow(latent_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f"Latent Forward Diffusion - Time Step {t_steps[idx]}", fontsize=12, fontweight="bold")
            ax.axis("off")

        # Save the latent progression plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Forward_Diffusion_Step.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
        
    print(f"Visualizations saved in '{save_dir}'")

# Define function to visualize reverse generation process 
def visualize_reverse_generation_co2_train(
    model_unet, 
    vqvae_model, 
    ddpm, 
    dataset,  # Dataset object instead of dataloader
    min_train_sg, 
    max_train_sg, 
    min_train_por, 
    max_train_por, 
    save_dir, 
    sample_indices,  # Indices of samples to fetch directly from the dataset
    steps
):
    """
    Perform reverse diffusion with VQ-VAE latent space and visualize results.

    Parameters:
    - model_unet: Conditional UNet model
    - vqvae_model: Pretrained VQ-VAE model
    - ddpm: Diffusion scheduler
    - dataset: Dataset object for fetching samples directly
    - min_train_sg, max_train_sg: Min/Max for CO₂ gas saturation denormalization
    - min_train_por, max_train_por: Min/Max for porosity denormalization
    - save_dir: Directory to save the visualizations
    - sample_indices: Indices of samples to visualize
    - steps: Reverse diffusion timesteps to collect
    """
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Get device from accelerator
    device = accelerator.device  

    # Load Pretrained Unet
    model_unet.to(device).eval() # Set to evaluation mode
    
    # Load Pretrained VQVAE
    vqvae_model.to(device).eval()  # Set to evaluation mode
    
    # Convert min and max lists to NumPy arrays
    min_train_sg = np.array(min_train_sg)
    max_train_sg = np.array(max_train_sg)
    
    # Convert min and max lists to NumPy arrays
    min_train_por = np.array(min_train_por)
    max_train_por = np.array(max_train_por)
    
    # Define grid for plotting (96x200 resolution)
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))

    # Fetch samples from dataset
    targets, labels = [], []
    for idx in sample_indices:
        target, label = dataset[idx]
        targets.append(target)
        labels.append(label)
    
    # Convert to tensors and move to device
    targets = torch.stack(targets).to(device)
    labels = torch.stack(labels).to(device)

    # Encode targets into latent space using VQ-VAE
    with torch.no_grad():
        latents, _ = vqvae_model.encode(targets)
    
    # --- START GENERATION (SAMPLING) TIMER ---
    gen_start_time = time.time()

    # Perform reverse diffusion in latent space
    final_latent_prediction, latent_collect = ddpm.ddpm_sampling(
        model=model_unet,
        num_samples=len(sample_indices),
        channels=vqvae_model.z_channels,
        img_size=(latents.shape[-2], latents.shape[-1]),
        labels=labels,
    )

    # Decode the final predictions back to the original space
    with torch.no_grad():
        final_predictions = vqvae_model.decode(final_latent_prediction)
    
    # --- END GENERATION (SAMPLING) TIMER ---
    gen_end_time = time.time()
    generation_time = gen_end_time - gen_start_time
    gen_hours = int(generation_time // 3600)
    gen_minutes = int((generation_time % 3600) // 60)
    gen_seconds = int(generation_time % 60)
        
    # Store metrics as lists 
    mae_list, mse_list, rmse_list, r2_list, mape_list, ssim_list = [], [], [], [], [], []

    # Plot and save results
    for i, (target, label, final_prediction) in enumerate(zip(targets, labels, final_predictions)):
        dataset_index = sample_indices[i]  # Use the actual dataset index for naming
        
        ### Only extract **Condition 2 (Porosity)**
        porosity_index = 2
        label = label[porosity_index, :, :]  # Only use condition index 2
        
        # Get per-sample min/max for porosity
        current_min_por = min_train_por[dataset_index]  
        current_max_por = max_train_por[dataset_index]
        
        label_denorm = denormalize_por_train(label.cpu().numpy(), np.array([current_min_por]), np.array([current_max_por]))
        
        # Mask non-zero porosity regions
        mask = label_denorm != 0
        thickness = mask[:, 0].sum()
        
        label_masked = label_denorm[mask].reshape((thickness, -1))
        
        # Denormalize target and prediction (CO₂ gas saturation)
        current_min_sg = min_train_sg[dataset_index]
        current_max_sg = max_train_sg[dataset_index]
        target_denorm = denormalize_co2_train(target.cpu().numpy(), np.array([current_min_sg]), np.array([current_max_sg]))
        target_masked = target_denorm[0, :, :][mask].reshape((thickness, -1))
        
        prediction_denorm = denormalize_co2_train(final_prediction.cpu().numpy(), np.array([current_min_sg]), np.array([current_max_sg]))
        prediction_masked = prediction_denorm[0, :, :][mask].reshape((thickness, -1))

        # Plot target, condition, and prediction
        fig_target, ax_target = plt.subplots(figsize=(10, 4))
        im_target = ax_target.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(target_masked),
            shading='auto', cmap='jet',
            vmin=target_masked.min(), vmax=target_masked.max()  # Dynamic range for original target
        )
        ax_target.set_title("Target - CO₂ Gas Saturation (-) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_target.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_target.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_target.set_xlim([0, 3500])
        fig_target.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Target.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_target)

        # Plot condition (porosity) separately
        fig_condition, ax_condition = plt.subplots(figsize=(10, 4))
        im_condition = ax_condition.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(label_masked), shading='auto', cmap='jet')
        ax_condition.set_title("Condition - Porosity ($\phi$)", fontsize=12, fontweight="bold")
        ax_condition.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_condition.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_condition.set_xlim([0, 100000])
        fig_condition.colorbar(im_condition, ax=ax_condition, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Condition.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_condition)

        # Plot prediction separately
        fig_prediction, ax_prediction = plt.subplots(figsize=(10, 4))
        im_prediction = ax_prediction.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(prediction_masked),
            shading='auto', cmap='jet',
            vmin=prediction_masked.min(), vmax=prediction_masked.max()  # Dynamic range for original target
        )
        ax_prediction.set_title("Prediction - CO₂ Gas Saturation (-) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_prediction.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_prediction.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_prediction.set_xlim([0, 3500])
        fig_prediction.colorbar(im_prediction, ax=ax_prediction, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Prediction.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_prediction)
        
        # Plot each reverse diffusion step in the same style as target and condition
        for step_idx, step in enumerate(reversed(steps)):  # Reverse the steps for correct order
            latent_step = latent_collect[len(steps) - step_idx - 1][i].cpu().numpy()  # Get reverse latent step
            latent_step_decoded = vqvae_model.decode(torch.tensor(latent_step).unsqueeze(0).to(accelerator.device)).squeeze(0).detach().cpu().numpy()
            latent_step_denorm = denormalize_co2_train(latent_step_decoded, np.array([current_min_sg]), np.array([current_max_sg]))
            latent_step_masked = latent_step_denorm[0, :, :][mask].reshape((thickness, -1))
            
            # Create a plot for the latent step
            fig, ax = plt.subplots(figsize=(10, 4))  # Same size as target and condition plots
            im = ax.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(latent_step_masked), cmap='jet', shading='auto', vmin=latent_step_masked.min(), vmax=latent_step_masked.max())
            ax.set_title(f"Reverse Step {step}", fontsize=12, fontweight='bold')  # Dynamic title with step number
            ax.set_xlabel("Radial Distance (m)", fontsize=10, fontweight='bold')
            ax.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight='bold')
            ax.set_xlim([0, 3500])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Save the plot for this step
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Reverse_Step_{step}.png"), dpi=600, bbox_inches='tight')
            plt.close(fig)
                
        # Plot absolute error between target and final prediction
        absolute_error = np.abs(target_masked - prediction_masked)
        
        # Compute metrics
        threshold = 1e-3  # Ignore values smaller than 1e-4
        valid_mask = target_masked > threshold
        mape = np.mean(np.abs((target_masked[valid_mask] - prediction_masked[valid_mask]) / target_masked[valid_mask])) * 100

        nonzero_mask = target_masked != 0  # Non-zero mask for all metrics
        mae = mean_absolute_error(target_masked[nonzero_mask], prediction_masked[nonzero_mask])
        mse = mean_squared_error(target_masked[nonzero_mask], prediction_masked[nonzero_mask])
        rmse = sqrt(mse)
        r2 = custom_r2_score(target_masked, prediction_masked)  
        ssim_value = ssim(
            target_masked, prediction_masked, data_range=target_masked.max() - target_masked.min(), win_size=3
        )
        
        # Append metrics to their respective lists
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        mape_list.append(mape)
        ssim_list.append(ssim_value)

        # Print metrics
        print(f"Sample {dataset_index}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  SSIM: {ssim_value:.4f}")
        
        # Scatter Plot for Original vs Predicted
        true_values = target_masked.flatten()
        predicted_values = prediction_masked.flatten()

        scatter_min = min(true_values.min(), predicted_values.min())
        scatter_max = max(true_values.max(), predicted_values.max())

        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
        ax_scatter.scatter(true_values, predicted_values, alpha=0.8, s=15, color="blue", label="year 30.0")
        ax_scatter.plot([scatter_min, scatter_max], [scatter_min, scatter_max], color="red", linestyle="--", label="Regression Line")
        ax_scatter.text(
            0.7, 0.1,
            f"$R^2$: {r2:.4f}",
            transform=ax_scatter.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax_scatter.set_xlabel("True CO₂ Gas Saturation (-)", fontsize=10, fontweight="bold")
        ax_scatter.set_ylabel("Predicted CO₂ Gas Saturation (-)", fontsize=10, fontweight="bold")
        ax_scatter.legend()
        ax_scatter.grid(linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Scatter_Sample_{dataset_index}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_scatter)
    
        # Plot absolute error and display metrics
        fig_error, ax_error = plt.subplots(figsize=(10, 4))
        im_error = ax_error.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(absolute_error), shading='auto', cmap="jet", vmin=0, vmax=0.25)
        ax_error.set_title("Absolute Error - CO₂ Gas Saturation (-) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_error.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_error.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_error.set_xlim([0, 3500])  # Match the x-axis limit
        
        # Add metrics as a text box
        metrics_text = (
            f"MAE: {mae:.4f}\n"
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"R²: {r2:.4f}\n"
            f"MAPE: {mape:.2f}%\n"
            f"SSIM: {ssim_value:.4f}"
        )
        ax_error.text(
            0.95, 0.05, metrics_text, transform=ax_error.transAxes, fontsize=16, 
            color="white", verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", edgecolor="white")
        )
        
        fig_error.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Absolute_Error.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_error)
    
    # Calculate average metrics across all samples 
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
    
    # Save average metrics values into txt file
    with open(os.path.join(save_dir, "diffusion_train_co2_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        # Add generation (sampling) time ONLY
        log_file.write("\nTotal Generation Time:\n")
        log_file.write(f"{gen_hours}h {gen_minutes}m {gen_seconds}s\n")

    print(f"\nTotal Generation Time: {gen_hours}h {gen_minutes}m {gen_seconds}s")
    print(f"Plots saved in '{save_dir}'")

# Define function to visualize reverse generation process 
def visualize_reverse_generation_dP_train(
    model_unet, 
    vqvae_model, 
    ddpm, 
    dataset, 
    min_train_dP, 
    max_train_dP, 
    min_train_por, 
    max_train_por, 
    save_dir,
    sample_indices, 
    steps
):
    """
    Perform reverse diffusion with VQ-VAE latent space and visualize results.

    Parameters:
    - model_unet: Conditional UNet model
    - vqvae_model: Pretrained VQ-VAE model
    - ddpm: Diffusion scheduler
    - dataloader: DataLoader for the dataset
    - min_train_sg, max_train_sg: Min/Max for CO₂ gas saturation denormalization
    - min_train_por, max_train_por: Min/Max for porosity denormalization
    - save_dir: Directory to save the visualizations
    - batch_number: Batch index to process
    - sample_indices: Indices of samples to visualize within the batch
    - steps: Reverse diffusion timesteps to collect
    """
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    
    # Instantiate Accelerator
    accelerator = Accelerator(split_batches=True)
    
    # Get device from accelerator
    device = accelerator.device 
    
    # Load Pretrained Unet
    model_unet.to(device).eval # Set to evaluation mode
    
    # Load Pretrained VQVAE
    vqvae_model.to(device).eval # Set to evaluation mode
    
    # Convert min and max lists to NumPy arrays
    min_train_dP = np.array(min_train_dP)
    max_train_dP = np.array(max_train_dP)
    
    # Convert min and max lists to NumPy arrays
    min_train_por = np.array(min_train_por)
    max_train_por = np.array(max_train_por)
    
    # Define grid for plotting (96x200 resolution)
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))

    # Fetch samples from dataset
    targets, labels = [], []
    for idx in sample_indices:
        target, label = dataset[idx]
        targets.append(target)
        labels.append(label)
    
    # Convert to tensors and move to device
    targets = torch.stack(targets).to(device)
    labels = torch.stack(labels).to(device)

    # Encode targets into latent space using VQ-VAE
    with torch.no_grad():
        latents, _ = vqvae_model.encode(targets)
    
    # --- START GENERATION (SAMPLING) TIMER ---
    gen_start_time = time.time()

    # Perform reverse diffusion in latent space
    final_latent_prediction, latent_collect = ddpm.ddpm_sampling(
        model=model_unet,
        num_samples=len(sample_indices),
        channels=vqvae_model.z_channels,
        img_size=(latents.shape[-2], latents.shape[-1]),
        labels=labels,
    )

    # Decode the final predictions back to the original space
    with torch.no_grad():
        final_predictions = vqvae_model.decode(final_latent_prediction)
    
    # --- END GENERATION (SAMPLING) TIMER ---
    gen_end_time = time.time()
    generation_time = gen_end_time - gen_start_time
    gen_hours = int(generation_time // 3600)
    gen_minutes = int((generation_time % 3600) // 60)
    gen_seconds = int(generation_time % 60)
        
    # Store metrics as lists 
    mae_list, mse_list, rmse_list, r2_list, mape_list, ssim_list = [], [], [], [], [], []

    # Plot and save results
    for i, (target, label, final_prediction) in enumerate(zip(targets, labels, final_predictions)):
        dataset_index = sample_indices[i]  # Use the actual dataset index for naming
        
        ### Only extract **Condition 2 (Porosity)**
        porosity_index = 2
        label = label[porosity_index, :, :]  # Only use condition index 2
        
        # Get per-sample min/max for porosity
        current_min_por = min_train_por[dataset_index]  
        current_max_por = max_train_por[dataset_index]

        label_denorm = denormalize_por_train(label.cpu().numpy(), np.array([current_min_por]), np.array([current_max_por]))
        
        # Mask non-zero porosity regions
        mask = label_denorm != 0
        thickness = mask[:, 0].sum()
        
        label_masked = label_denorm[mask].reshape((thickness, -1))
        
        # Denormalize target and prediction 
        current_min_dP = min_train_dP[dataset_index]
        current_max_dP = max_train_dP[dataset_index]
        target_denorm = denormalize_dP_train(target.cpu().numpy(), np.array([current_min_dP]), np.array([current_max_dP]))
        target_masked = target_denorm[0, :, :][mask].reshape((thickness, -1))

        prediction_denorm = denormalize_dP_train(final_prediction.cpu().numpy(), np.array([current_min_dP]), np.array([current_max_dP]))
        prediction_masked = prediction_denorm[0, :, :][mask].reshape((thickness, -1))

        # Plot target, condition, and prediction
        fig_target, ax_target = plt.subplots(figsize=(10, 4))
        im_target = ax_target.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(target_masked),
            shading='auto', cmap='jet',
            vmin=target_masked.min(), vmax=target_masked.max()  # Dynamic range for original target
        )
        ax_target.set_title("Target - Pressure Build-Up (bar) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_target.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_target.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_target.set_xlim([0, 3500])
        fig_target.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Target.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_target)

        # Plot condition (porosity) separately
        fig_condition, ax_condition = plt.subplots(figsize=(10, 4))
        im_condition = ax_condition.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(label_masked), shading='auto', cmap='jet')
        ax_condition.set_title("Condition - Porosity ($\phi$)", fontsize=12, fontweight="bold")
        ax_condition.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_condition.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_condition.set_xlim([0, 100000])
        fig_condition.colorbar(im_condition, ax=ax_condition, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Condition.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_condition)

        # Plot prediction separately
        fig_prediction, ax_prediction = plt.subplots(figsize=(10, 4))
        im_prediction = ax_prediction.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(prediction_masked),
            shading='auto', cmap='jet',
            vmin=prediction_masked.min(), vmax=prediction_masked.max()  # Dynamic range for original target
        )
        ax_prediction.set_title("Prediction - Pressure Build-Up (bar) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_prediction.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_prediction.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_prediction.set_xlim([0, 3500])
        fig_prediction.colorbar(im_prediction, ax=ax_prediction, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Prediction.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_prediction)
        
        # Plot reverse diffusion process
        for step_idx, step in enumerate(reversed(steps)):  # Reverse the steps for correct order
            latent_step = latent_collect[len(steps) - step_idx - 1][i].cpu().numpy()  # Get reverse latent step
            latent_step_decoded = vqvae_model.decode(torch.tensor(latent_step).unsqueeze(0).to(accelerator.device)).squeeze(0).detach().cpu().numpy()
            latent_step_denorm = denormalize_dP_train(latent_step_decoded, np.array([current_min_dP]), np.array([current_max_dP]))
            latent_step_masked = latent_step_denorm[0, :, :][mask].reshape((thickness, -1))
            
            # Create a plot for the latent step
            fig, ax = plt.subplots(figsize=(10, 4))  # Same size as target and condition plots
            im = ax.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(latent_step_masked), cmap='jet', shading='auto', vmin=latent_step_masked.min(), vmax=latent_step_masked.max())
            ax.set_title(f"Reverse Step {step}", fontsize=12, fontweight='bold')  # Dynamic title with step number
            ax.set_xlabel("Radial Distance (m)", fontsize=10, fontweight='bold')
            ax.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight='bold')
            ax.set_xlim([0, 3500])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Save the plot for this step
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Reverse_Step_{step}.png"), dpi=600, bbox_inches='tight')
            plt.close(fig)
        
        # Plot absolute error between target and final prediction
        absolute_error = np.abs(target_masked - prediction_masked)
        
        # Compute metrics
        threshold = 1e-3  # Ignore values smaller than 1e-4
        valid_mask = target_masked > threshold
        mape = np.mean(np.abs((target_masked[valid_mask] - prediction_masked[valid_mask]) / target_masked[valid_mask])) * 100

        nonzero_mask = target_masked != 0  # Non-zero mask for all metrics
        mae = mean_absolute_error(target_masked[nonzero_mask], prediction_masked[nonzero_mask])
        mse = mean_squared_error(target_masked[nonzero_mask], prediction_masked[nonzero_mask])
        rmse = sqrt(mse)
        r2 = custom_r2_score(target_masked, prediction_masked)  
        
        ssim_value = ssim(
            target_masked, prediction_masked, data_range=target_masked.max() - target_masked.min(), win_size=3
        )
        
        # Append metrics to their respective lists
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        mape_list.append(mape)
        ssim_list.append(ssim_value)

        # Print metrics
        print(f"Sample {dataset_index}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  SSIM: {ssim_value:.4f}")
        
        # Scatter Plot for Original vs Predicted
        true_values = target_masked.flatten()
        predicted_values = prediction_masked.flatten()

        scatter_min = min(true_values.min(), predicted_values.min())
        scatter_max = max(true_values.max(), predicted_values.max())

        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
        ax_scatter.scatter(true_values, predicted_values, alpha=0.8, s=15, color="blue", label="year 30.0")
        ax_scatter.plot([scatter_min, scatter_max], [scatter_min, scatter_max], color="red", linestyle="--", label="Regression Line")
        ax_scatter.text(
            0.7, 0.1,
            f"$R^2$: {r2:.4f}",
            transform=ax_scatter.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax_scatter.set_xlabel("True Pressure Build-Up (bar)", fontsize=10, fontweight="bold")
        ax_scatter.set_ylabel("Predicted Pressure Build-Up (bar)", fontsize=10, fontweight="bold")
        ax_scatter.legend()
        ax_scatter.grid(linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Scatter_Sample_{dataset_index}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_scatter)
        
        # Plot absolute error and display metrics
        fig_error, ax_error = plt.subplots(figsize=(10, 4))
        im_error = ax_error.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(absolute_error), shading='auto', cmap="jet")
        ax_error.set_title("Absolute Error - Pressure Build-Up (bar) @ t = 30.0 y", fontsize=12, fontweight="bold")
        ax_error.set_xlabel("Radial Distance (m)", fontsize=10, fontweight="bold")
        ax_error.set_ylabel("Reservoir Thickness (m)", fontsize=10, fontweight="bold")
        ax_error.set_xlim([0, 3500])  # Match the x-axis limit

        # Add metrics as a text box
        metrics_text = (
            f"MAE: {mae:.4f}\n"
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"R²: {r2:.4f}\n"
            f"MAPE: {mape:.2f}%\n"
            f"SSIM: {ssim_value:.4f}"
        )
        ax_error.text(
            0.95, 0.05, metrics_text, transform=ax_error.transAxes, fontsize=16, 
            color="white", verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", edgecolor="white")
        )

        fig_error.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Sample_{dataset_index}_Absolute_Error.png"), dpi=600, bbox_inches="tight")
        plt.close(fig_error)
    
    # Calculate average metrics across all samples 
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
    
    # Save average metrics values into txt file
    with open(os.path.join(save_dir, "diffusion_train_dP_average_metrics.txt"), "w") as log_file:
        log_file.write("====== AVERAGE METRICS ACROSS ALL SAMPLES ======\n")
        log_file.write(f"Average MAE: {avg_mae:.4f}\n")
        log_file.write(f"Average MSE: {avg_mse:.4f}\n")
        log_file.write(f"Average RMSE: {avg_rmse:.4f}\n")
        log_file.write(f"Average R²: {avg_r2:.4f}\n")
        log_file.write(f"Average MAPE: {avg_mape:.2f}%\n")
        log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        # Add generation (sampling) time ONLY
        log_file.write("\nTotal Generation Time:\n")
        log_file.write(f"{gen_hours}h {gen_minutes}m {gen_seconds}s\n")

    print(f"\nTotal Generation Time: {gen_hours}h {gen_minutes}m {gen_seconds}s")
    print(f"Plots saved in '{save_dir}'")
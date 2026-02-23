###################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: This script processes, visualizes, and analyzes high-dimensional reservoir simulation datasets.
# 
###################################################################

# Import libraries 
import os
import csv
import torch
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

# Set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define function to process and visualize the dataset -> resolution 96*200
def dataset_samples_visualization(data_dir, parameter_type, dataset_type, output_dir, device=device, unit='-'):
    """
    Processes and visualizes a reservoir simulation dataset with resolution 96x200. 
    The function generates and saves visualizations of input parameter maps, 
    scalar parameters, and outputs (CO₂ gas saturation or pressure buildup) 
    for each sample across time frames.

    Args:
        data_dir (str): Path to the directory containing input/output dataset files.
        parameter_type (str): Type of output parameter ('sg' for gas saturation, 'dP' for pressure buildup).
        dataset_type (str): Dataset category ('training', 'validation', or 'test').
        output_dir (str): Directory to save generated visualizations.
        device (torch.device): Device to use ('cpu' or 'cuda').
        unit (str): Unit of the output parameter (default is '-').
    """
    # Load the data
    inputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_inputs.pt')
    outputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_outputs.pt')
    print(f"{parameter_type} {dataset_type} inputs shape:", inputs.shape)
    print(f"{parameter_type} {dataset_type} outputs shape:", outputs.shape)
    
    # Prepare the data loader
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, outputs), 
        batch_size=1, 
        shuffle=False
    )

    # Define grid for plotting for 96*200 resolution 
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))
    
    # Helper function to plot data on a color map
    def pcolor(x, xlim=None):
        plt.jet()
        plt.xlabel("Radial Distance (m)", fontsize=10, fontweight='bold')
        plt.ylabel("Reservoir Thickness (m)", fontsize=10, fontweight='bold')
        cax = plt.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(x), shading='auto')
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim([0, np.max(X)])
        colorbar = plt.colorbar(cax, fraction=0.03, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)
    
    # Define denormalization function for dP
    def dnorm_dP(dP):
        dP = dP * 18.772821433027488
        dP = dP + 4.172939172019009 
        return dP

    # Calculate cumulative time intervals
    times = np.cumsum(np.power(1.421245, range(24)))
    time_print = []
    for t in range(times.shape[0]):
        if times[t] < 365:
            title = str(int(times[t])) + ' d'
            time_print.append(title)
        else:
            title = f'{round(int(times[t]) / 365, 1)} y'
            time_print.append(title)

    # Print the generated time labels to confirm
    print("Generated time labels:", time_print)

    # Define de-normalization functions
    dnorm_inj = lambda a: (a * (3e6 - 3e5) + 3e5) / (1e6 / 365 * 1000 / 1.862)
    dnorm_temp = lambda a: a * (180 - 30) + 30
    dnorm_P = lambda a: a * (300 - 100) + 100
    dnorm_lam = lambda a: a * 0.4 + 0.3
    dnorm_Swi = lambda a: a * 0.2 + 0.1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all realizations
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_plot = y.cpu().detach().numpy()
        x_plot = x.cpu().detach().numpy()
        
        # Determine the reservoir thickness
        mask = x_plot[0, :, :, 0, 0] != 0 
        thickness = sum(mask[:, 0])
    
        # Extract input parameter maps with masking
        kr_map = np.exp(x_plot[0, :, :, 0, 0][mask].reshape((thickness, -1)) * 15)
        kz_map = np.exp(x_plot[0, :, :, 0, 1][mask].reshape((thickness, -1)) * 15)
        poro_map = x_plot[0, :, :, 0, 2][mask].reshape((thickness, -1))
        perf_map = x_plot[0, :, :, 0, 3][mask].reshape((thickness, -1))

        # De-normalize scalar parameters
        inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
        temperature = dnorm_temp(x_plot[0, 0, 0, 0, 5])
        pressure = dnorm_P(x_plot[0, 0, 0, 0, 6])
        Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
        lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])

        # Print sample processing information
        print(f'Sample {i+1} - Injection Rate: {inj_rate:.2f} MT/yr, Isothermal Reservoir Temperature: {temperature:.1f} °C, '
              f'Initial Reservoir Pressure: {pressure:.1f} bar, Irreducible Water Saturation: {Swi:.2f}, Van Genuchten Scaling Factor: {lam:.2f}')

        # Save each input parameter
        parameter_maps = {
            "kr": (kr_map, "$k_r$ (mD)", [0, 100000]),
            "kz": (kz_map, "$k_z$ (mD)", [0, 100000]),
            "phi": (poro_map, "$\phi$ (-)", [0, 100000]),
            "perf": (perf_map, "Perforation", [0, 2000])
        }
        
        for param_name, (param_map, title, xlim) in parameter_maps.items():
            plt.figure(figsize=(10, 4))
            pcolor(param_map, xlim=xlim)
            plt.title(title, fontweight='bold', fontsize=12)
            param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_{param_name}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, param_filename), dpi=600, bbox_inches='tight',
                        #transparent=True
                       )
            plt.close()

        # Save scalar parameters
        plt.figure(figsize=(6, 3))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Injection Rate ($Q$) = {inj_rate:.2f} MT/yr\nIsothermal Reservoir Temperature ($T$) = {temperature:.1f} °C\n'
                            f'Initial Reservoir Pressure ($P_{{init}}$) = {pressure:.1f} bar\nIrreducible Water Saturation ($S_{{wi}}$) = {Swi:.2f}\nVan Genuchten Scaling Factor ($\\lambda$) =  {lam:.2f}', ha='center', va='center', fontsize=10)
        scalar_param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_Scalar_Parameters.png'
        plt.savefig(os.path.join(output_dir, scalar_param_filename), dpi=600, bbox_inches='tight',
                    #transparent=True
                    )
        plt.close()

        # Loop through each time frame and save CO₂ saturation and Pressure BuildUp in separate files
        for t in range(y_plot.shape[3]):
            plt.figure(figsize=(10, 4))
            
            # Apply denormalization if plotting pressure
            if parameter_type == 'dP':
                denormalized_data = dnorm_dP(y_plot[0, :, :, t])
                pcolor(denormalized_data[mask].reshape((thickness, -1)))
            else:
                pcolor(y_plot[0, :, :, t][mask].reshape((thickness, -1)))

            if parameter_type == 'sg':
                plt.title(f'CO₂ Gas Saturation ({unit}) @ t = {time_print[t]}', fontweight='bold')
            elif parameter_type == 'dP':
                plt.title(f'Pressure Build-Up ({unit}) @ t = {time_print[t]}', fontweight='bold')
            plt.xlim([0, 3500])
            
            ## First Option: Save plots with colorbar and labels 
            plot_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_time_{time_print[t]}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=600, bbox_inches='tight',
                        #transparent=True 
                        )
            plt.close()

    print(f'All plots saved in {output_dir}')
    
# Define function to process and visualize the dataset -> resolution 64*64 (in case we do not rely on the latent space)
def dataset_samples_visualization_64x64(data_dir, parameter_type, dataset_type, output_dir, device=device, unit='-'):
    """
    Processes and visualizes reservoir simulation data with spatial resolution 64x64.
    The function generates and saves:
        - Input parameter maps (e.g., permeability, porosity).
        - Scalar parameters as textual plots.
        - Output maps (CO₂ gas saturation or pressure buildup) for all time frames.

    Args:
        data_dir (str): Path to the directory containing input and output `.pt` files.
        parameter_type (str): Type of output parameter ('sg' for CO₂ gas saturation, 'dP' for pressure buildup).
        dataset_type (str): Dataset category ('training', 'validation', or 'test').
        output_dir (str): Directory where generated plots will be saved.
        device (str): Computation device ('cpu' or 'cuda'). Defaults to device in use. 
        unit (str): Unit for the output parameter (default is '-').

    Outputs:
        - Parameter maps: Visualizations for kr (radial permeability), kz (vertical permeability), 
          porosity (phi), and perforation maps.
        - Scalar parameters: De-normalized values for injection rate, reservoir temperature, 
          initial pressure, irreducible water saturation, and Van Genuchten scaling factor.
        - Time-dependent outputs: CO₂ saturation or pressure buildup for each time step.
    
    Notes:
        - Data files must follow the naming convention: 
          `{parameter_type}_{dataset_type}_inputs.pt` and `{parameter_type}_{dataset_type}_outputs.pt`.
        - Outputs are saved as high-resolution `.png` files in the specified output directory.
    """
    # Load the data
    inputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_inputs.pt')
    outputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_outputs.pt')
    
    # Perform slicing to change spatial dimensions to [64, 64]
    inputs = inputs[:, :64, :64, :]  
    outputs = outputs[:, :64, :64, :]  
    
    # Print data shape 
    print(f"{parameter_type} {dataset_type} inputs shape:", inputs.shape)
    print(f"{parameter_type} {dataset_type} outputs shape:", outputs.shape)

    # Prepare the data loader for outputs only
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, outputs), 
        batch_size=1, 
        shuffle=False
    )
    
    # Define denormalization function for dP
    def dnorm_dP(dP):
        dP = dP * 18.772821433027488
        dP = dP + 4.172939172019009 
        return dP
    
    # Calculate cumulative time intervals
    times = np.cumsum(np.power(1.421245, range(24)))
    time_print = []
    for t in range(times.shape[0]):
        if times[t] < 365:
            title = str(int(times[t])) + ' d'
            time_print.append(title)
        else:
            title = f'{round(int(times[t]) / 365, 1)} y'
            time_print.append(title)

    # Print the generated time labels to confirm
    print("Generated time labels:", time_print)
    
    # Define de-normalization functions
    dnorm_inj = lambda a: (a * (3e6 - 3e5) + 3e5) / (1e6 / 365 * 1000 / 1.862)
    dnorm_temp = lambda a: a * (180 - 30) + 30
    dnorm_P = lambda a: a * (300 - 100) + 100
    dnorm_lam = lambda a: a * 0.4 + 0.3
    dnorm_Swi = lambda a: a * 0.2 + 0.1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all samples in the data loader
    for i, (x, y) in enumerate(data_loader):
        print(f"Processing sample {i+1} of {len(data_loader)}")  # Print the sample number
        x, y = x.to(device), y.to(device)
        y_plot = y.cpu().detach().numpy()
        x_plot = x.cpu().detach().numpy()
        
        # Apply mask to focus on relevant data points
        mask = x_plot[0, :, :, 0, 0] != 0 
        thickness = sum(mask[:, 0])
        
        # Extract input parameter maps with masking
        kr_map = np.exp(x_plot[0, :, :, 0, 0][mask].reshape((thickness, -1)) * 15)
        kz_map = np.exp(x_plot[0, :, :, 0, 1][mask].reshape((thickness, -1)) * 15)
        poro_map = x_plot[0, :, :, 0, 2][mask].reshape((thickness, -1))
        perf_map = x_plot[0, :, :, 0, 3][mask].reshape((thickness, -1))
        
        # De-normalize scalar parameters
        inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
        temperature = dnorm_temp(x_plot[0, 0, 0, 0, 5])
        pressure = dnorm_P(x_plot[0, 0, 0, 0, 6])
        Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
        lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])
        
        # Print sample processing information
        print(f'Sample {i+1} - Injection Rate: {inj_rate:.2f} MT/yr, Isothermal Reservoir Temperature: {temperature:.1f} °C, '
              f'Initial Reservoir Pressure: {pressure:.1f} bar, Irreducible Water Saturation: {Swi:.2f}, Van Genuchten Scaling Factor: {lam:.2f}')
        
        # Save each input parameter
        parameter_maps = {
            "kr": (kr_map, "$k_r$ (mD)"),
            "kz": (kz_map, "$k_z$ (mD)"),
            "phi": (poro_map, "$\phi$ (-)"),
            "perf": (perf_map, "Perforation")
        }
        
        # Plot each parameter map using imshow
        for param_name, (param_map, title) in parameter_maps.items():
            plt.figure(figsize=(6, 6))
            if param_name == "perf":
                plt.imshow(param_map, cmap='jet', aspect='auto',  vmin=0, vmax=1)  # Adjust `vmin` and `vmax` as necessary
            else:
                plt.imshow(param_map, cmap='jet', aspect='auto') 
            plt.colorbar()
            plt.title(title, fontweight='bold', fontsize=12)
            plt.axis('off')
            param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_{param_name}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, param_filename), dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # Save scalar parameters
        plt.figure(figsize=(6, 3))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Injection Rate ($Q$) = {inj_rate:.2f} MT/yr\nIsothermal Reservoir Temperature ($T$) = {temperature:.1f} °C\n'
                            f'Initial Reservoir Pressure ($P_{{init}}$) = {pressure:.1f} bar\nIrreducible Water Saturation ($S_{{wi}}$) = {Swi:.2f}\nVan Genuchten Scaling Factor ($\\lambda$) =  {lam:.2f}', ha='center', va='center', fontsize=10)
        scalar_param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_Scalar_Parameters.png'
        plt.savefig(os.path.join(output_dir, scalar_param_filename), dpi=600, bbox_inches='tight')
        plt.close()

        # Loop through each time frame to plot and save CO₂ saturation
        for t in range(y_plot.shape[3]):
            plt.figure(figsize=(9, 9))
            if parameter_type == 'dP':
                plt.imshow(dnorm_dP(y_plot[0, :, :, t][mask].reshape((thickness, -1))), cmap='jet', aspect='auto')  
                plt.title(f'Pressure Build-Up ({unit}) @ t = {time_print[t]}', fontweight='bold')
            else:
                plt.imshow(y_plot[0, :, :, t][mask].reshape((thickness, -1)), cmap='jet', aspect='auto', vmin=0, vmax=1)  # Set colorbar range from 0 to 1
                plt.title(f'CO₂ Gas Saturation ({unit}) @ t = {time_print[t]}', fontweight='bold')
            plt.colorbar()
            plt.axis('off') 

            # Save plot
            plot_filename =  f'{dataset_type}_{parameter_type}_sample_{i+1}_time_{time_print[t]}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f'All plots saved in {output_dir}')

# Define function to process and visualize the dataset without labels -> resolution 96*200
def dataset_samples_visualization_no_labels(data_dir, parameter_type, dataset_type, output_dir, device=device, unit='-'):
    """
    Processes and visualizes reservoir simulation data with resolution 96x200, 
    focusing on input and output parameter maps **without labels**. 
    The function generates and saves:
        - Input parameter maps: kr (radial permeability), kz (vertical permeability), 
          porosity, and perforation.
        - Scalar parameters: Injection rate, reservoir temperature, 
          initial pressure, irreducible water saturation, and Van Genuchten scaling factor.
        - Time-dependent outputs: CO₂ gas saturation or pressure buildup maps 
          (denormalized where necessary).

    Args:
        data_dir (str): Path to the directory containing input and output `.pt` files.
        parameter_type (str): Type of output parameter ('sg' for CO₂ gas saturation, 'dP' for pressure buildup).
        dataset_type (str): Dataset category ('training', 'validation', or 'test').
        output_dir (str): Directory where generated plots will be saved.
        device (torch.device): Computation device ('cpu' or 'cuda'). Defaults to device in use.
        unit (str): Unit for the output parameter (default is '-').

    Outputs:
        - Input Parameter Maps: Visualizations of permeability, porosity, and perforation.
        - Scalar Parameters: De-normalized reservoir parameters saved as text visualizations.
        - Output Maps: Visualizations of CO₂ gas saturation or pressure buildup 
          over multiple time steps, saved without colorbars or labels.

    Notes:
        - Data files must follow the naming convention: 
          `{parameter_type}_{dataset_type}_inputs.pt` and `{parameter_type}_{dataset_type}_outputs.pt`.
        - Outputs are saved as `.png` files with high resolution.
        - Colorbars and axes are removed for time-dependent output visualizations.
    """
    # Load the data
    inputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_inputs.pt')
    outputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_outputs.pt')
    print(f"{parameter_type} {dataset_type} inputs shape:", inputs.shape)
    print(f"{parameter_type} {dataset_type} outputs shape:", outputs.shape)

    # Prepare the data loader
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, outputs), 
        batch_size=1, 
        shuffle=False
    )

    # Define grid for plotting
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))

    def pcolor(x, xlim=None):
        plt.jet()
        plt.xlabel("Radial Distance (m)", fontsize=10, fontweight='bold')
        plt.ylabel("Reservoir Thickness (m)", fontsize=10, fontweight='bold')
        cax = plt.pcolor(X[:thickness, :], Y[:thickness, :], np.flipud(x), shading='auto')
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim([0, np.max(X)])
        #colorbar = plt.colorbar(cax, fraction=0.03, pad=0.04)
        #colorbar.ax.tick_params(labelsize=8)
    
    # Define denormalization function for dP
    def dnorm_dP(dP):
        dP = dP * 18.772821433027488
        dP = dP + 4.172939172019009 
        return dP

    # Calculate cumulative time intervals
    times = np.cumsum(np.power(1.421245, range(24)))
    time_print = []
    for t in range(times.shape[0]):
        if times[t] < 365:
            title = str(int(times[t])) + ' d'
            time_print.append(title)
        else:
            title = f'{round(int(times[t]) / 365, 1)} y'
            time_print.append(title)

    # Print the generated time labels to confirm
    print("Generated time labels:", time_print)

    # Define de-normalization functions
    dnorm_inj = lambda a: (a * (3e6 - 3e5) + 3e5) / (1e6 / 365 * 1000 / 1.862)
    dnorm_temp = lambda a: a * (180 - 30) + 30
    dnorm_P = lambda a: a * (300 - 100) + 100
    dnorm_lam = lambda a: a * 0.4 + 0.3
    dnorm_Swi = lambda a: a * 0.2 + 0.1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all realizations
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_plot = y.cpu().detach().numpy()
        x_plot = x.cpu().detach().numpy()
        
        # Determine the reservoir thickness
        mask = x_plot[0, :, :, 0, 0] != 0 
        thickness = sum(mask[:, 0])

        # Extract input parameter maps with masking
        kr_map = np.exp(x_plot[0, :, :, 0, 0][mask].reshape((thickness, -1)) * 15)
        kz_map = np.exp(x_plot[0, :, :, 0, 1][mask].reshape((thickness, -1)) * 15)
        poro_map = x_plot[0, :, :, 0, 2][mask].reshape((thickness, -1))
        perf_map = x_plot[0, :, :, 0, 3][mask].reshape((thickness, -1))

        # De-normalize scalar parameters
        inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
        temperature = dnorm_temp(x_plot[0, 0, 0, 0, 5])
        pressure = dnorm_P(x_plot[0, 0, 0, 0, 6])
        Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
        lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])

        # Print sample processing information
        print(f'Sample {i+1} - Injection Rate: {inj_rate:.2f} MT/yr, Isothermal Reservoir Temperature: {temperature:.1f} °C, '
              f'Initial Reservoir Pressure: {pressure:.1f} bar, Irreducible Water Saturation: {Swi:.2f}, Van Genuchten Scaling Factor: {lam:.2f}')

        # Save each input parameter
        parameter_maps = {
            "kr": (kr_map, "$k_r$ (mD)", [0, 100000]),
            "kz": (kz_map, "$k_z$ (mD)", [0, 100000]),
            "phi": (poro_map, "$\phi$ (-)", [0, 100000]),
            "perf": (perf_map, "Perforation", [0, 2000])
        }
        
        for param_name, (param_map, title, xlim) in parameter_maps.items():
            plt.figure(figsize=(10, 4))
            pcolor(param_map, xlim=xlim)
            plt.title(title, fontweight='bold', fontsize=12)
            param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_{param_name}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, param_filename), dpi=600, bbox_inches='tight',
                        #transparent=True
                       )
            plt.close()

        # Save scalar parameters
        plt.figure(figsize=(6, 3))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Injection Rate ($Q$) = {inj_rate:.2f} MT/yr\nIsothermal Reservoir Temperature ($T$) = {temperature:.1f} °C\n'
                            f'Initial Reservoir Pressure ($P_{{init}}$) = {pressure:.1f} bar\nIrreducible Water Saturation ($S_{{wi}}$) = {Swi:.2f}\nVan Genuchten Scaling Factor ($\\lambda$) =  {lam:.2f}', ha='center', va='center', fontsize=10)
        scalar_param_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_Scalar_Parameters.png'
        plt.savefig(os.path.join(output_dir, scalar_param_filename), dpi=600, bbox_inches='tight',
                    #transparent=True
                    )
        plt.close()

        # Loop through each time frame and save CO₂ saturation and Pressure BuildUp in separate files
        for t in range(y_plot.shape[3]):
            plt.figure(figsize=(10, 4))
            # Apply denormalization if plotting pressure
            if parameter_type == 'dP':
                pcolor(dnorm_dP(y_plot[0, :, :, t])[mask].reshape((thickness, -1)))
            else:
                pcolor(y_plot[0, :, :, t][mask].reshape((thickness, -1)))

            #if parameter_type == 'sg':
                #plt.title(f'CO₂ Gas Saturation ({unit}) @ t = {time_print[t]}', fontweight='bold')
            #elif parameter_type == 'dP':
                #plt.title(f'Pressure Build-Up ({unit}) @ t = {time_print[t]}', fontweight='bold')
            plt.xlim([0, 3500])
            
            ## Second Option: Save plots without colorbar and labels 
            # Turn off axis
            plt.axis('off')
            # Save the plot 
            plot_filename = f'{dataset_type}_{parameter_type}_sample_{i+1}_time_{time_print[t]}.png'
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f'All plots saved in {output_dir}')

# Define function to save input scalar parameters into CSV files
def save_scalar_parameters_to_csv(data_dir, parameter_type, dataset_type, output_dir, device=device):
    """
    Extracts and saves scalar input parameters (e.g., injection rate, temperature, pressure)
    into a CSV file for a given reservoir simulation dataset.

    Args:
        data_dir (str): Path to the directory containing input and output `.pt` files.
        parameter_type (str): Type of output parameter ('sg' for CO₂ gas saturation, 'dP' for pressure buildup).
        dataset_type (str): Dataset category ('training', 'validation', or 'test').
        output_dir (str): Directory where the scalar parameters CSV file will be saved.
        device (torch.device): Computation device ('cpu' or 'cuda'). Defaults to the device in use.

    Outputs:
        - A CSV file containing scalar parameters for each sample:
          Columns include:
            - Sample index
            - Injection Rate (MT/yr)
            - Temperature (°C)
            - Initial Pressure (bar)
            - Irreducible Water Saturation
            - Van Genuchten Scaling Factor
    
    Notes:
        - Data files must follow the naming convention: 
          `{parameter_type}_{dataset_type}_inputs.pt` and `{parameter_type}_{dataset_type}_outputs.pt`.
        - The CSV file is saved in the specified `output_dir` with the name:
          `{parameter_type}_{dataset_type}_scalar_parameters.csv`.
    """
    # Load the data
    inputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_inputs.pt')
    outputs = torch.load(f'{data_dir}/{parameter_type}_{dataset_type}_outputs.pt')
    print(f"{parameter_type} {dataset_type} inputs shape:", inputs.shape)
    print(f"{parameter_type} {dataset_type} outputs shape:", outputs.shape)

    # Prepare the data loader
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, outputs), 
        batch_size=1, 
        shuffle=False
    )

    # Define de-normalization functions
    dnorm_inj = lambda a: (a * (3e6 - 3e5) + 3e5) / (1e6 / 365 * 1000 / 1.862)
    dnorm_temp = lambda a: a * (180 - 30) + 30
    dnorm_P = lambda a: a * (300 - 100) + 100
    dnorm_lam = lambda a: a * 0.4 + 0.3
    dnorm_Swi = lambda a: a * 0.2 + 0.1

    # Create output directory and CSV file
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{output_dir}/{parameter_type}_{dataset_type}_scalar_parameters.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sample", "Injection Rate (MT/yr)", "Temperature (°C)", "Initial Pressure (bar)", 
                         "Irreducible Water Saturation", "Van Genuchten Scaling Factor"])

    # Extract scalar parameters and save to CSV
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        x_plot = x.cpu().detach().numpy()

        # De-normalize scalar parameters
        inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
        temperature = dnorm_temp(x_plot[0, 0, 0, 0, 5])
        pressure = dnorm_P(x_plot[0, 0, 0, 0, 6])
        Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
        lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])

        # Append data to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, inj_rate, temperature, pressure, Swi, lam])

        print(f'Sample {i+1} - Injection Rate: {inj_rate:.2f} MT/yr, Temperature: {temperature:.1f} °C, '
              f'Pressure: {pressure:.1f} bar, Irreducible Water Saturation: {Swi:.2f}, Scaling Factor: {lam:.2f}')

    print(f"Scalar parameters saved in {csv_filename}")

# Define function to save KDE plots for each parameter
def save_kde_plots_from_csv(csv_file_path, output_dir, parameters):
    """
    Generates and saves KDE plots for specified parameters in a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing scalar parameters.
        output_dir (str): Directory to save the KDE plots.
        parameters (list of str): List of parameter names (as they appear in the CSV file) to plot.
    """
    # Define a custom fluorescent (neon) color palette
    fluo_colors = [
        "#39FF14",  # Neon green
        "#FF073A",  # Bright neon red
        "#00FFFF",  # Cyan
        "#FFAA1D",  # Neon orange
        "#FAED27",  # Bright yellow
        "#FF1493",  # Hot pink
        "#8A2BE2",  # Electric purple
        "#00FF7F",  # Spring green
        "#FF4500",  # Bright red-orange
        "#7FFF00"   # Chartreuse
    ]

    fluo_colors_cycle = cycle(fluo_colors)  # Cycle through colors if parameters exceed the color list
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the specified parameters and assign a unique fluorescent color
    for param, color in zip(parameters, fluo_colors_cycle):
        # Calculate the actual min and max values for the parameter
        min_val = data[param].min()
        max_val = data[param].max()

        # Create the KDE plot with a unique fluorescent color for the curve
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data[param], fill=True, color=color)
        plt.xlim(min_val, max_val)
        plt.xlabel(f"{param}", fontsize=10, fontweight='bold')
        plt.ylabel("Density", fontsize=10, fontweight='bold')
        plt.title(f"Kernel Density Estimation (KDE) - {param}", fontsize=12, fontweight='bold')

        # Prepare the output path with a safe filename
        safe_param_name = param.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_param_name}_kde_plot.png")
        print(f"Saving plot for {param} at {output_path}")

        # Save the plot as a PNG file
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()  

    print(f"KDE plots saved in {output_dir}")

# Define function to create GIF 
def create_parameter_gif(base_dir, dataset_type, parameter_type, sample_index, time_labels, duration=100):
    """
    Creates an animated GIF for a specified parameter (e.g., gas saturation or pressure buildup) 
    across all time frames for a given sample.

    Args:
        base_dir (str): 
            Base directory containing the parameter folders (e.g., 'samples_training_dataset_gas_saturation').
        dataset_type (str): 
            Type of dataset to process. Options include 'training', 'validation', or 'test'.
        parameter_type (str): 
            Parameter to visualize. Options include 'gas_saturation' (sg) or 'pressure_buildup' (dP).
        sample_index (int): 
            Index of the sample to process (0-based index; e.g., 0 corresponds to sample 1).
        time_labels (list of str): 
            List of time labels corresponding to each time frame (e.g., ['1 d', '2 d', ..., '1 y']).
        duration (int): 
            Duration of each frame in the GIF in milliseconds. Default is 100ms per frame.

    Outputs:
        - An animated GIF showing the progression of the specified parameter over time 
          for the selected sample.
    
    Notes:
        - The input image files must follow a specific naming convention:
          `{image_prefix}_{parameter_abbr}_sample_{sample_index + 1}_time_{time_label}.png`
          where `image_prefix` is determined by the dataset type:
            - 'train' for training
            - 'val' for validation
            - 'test' for test
        - Output GIFs are saved in a directory named 'GIFs_gas_saturation' (created automatically if it doesn't exist).

    Example:
        To create a GIF for gas saturation (sg) for Sample 1 in the training dataset:
        ```python
        create_parameter_gif(
            base_dir='path/to/base_dir',
            dataset_type='training',
            parameter_type='gas_saturation',
            sample_index=0,
            time_labels=['1 d', '10 d', '30 d'],
            duration=100
        )
        ```
        Output GIF will be saved as:
        `path/to/base_dir/GIFs_gas_saturation/training_gas_saturation_sample_1.gif`
    """
    # Map dataset types to image prefixes for filenames
    image_prefix_map = {
        'training': 'train',
        'validation': 'val',
        'test': 'test'
    }
    
    # Use `training`, `validation`, `test` for folder names and `train`, `val`, `test` for file prefixes
    folder_name = dataset_type  # Folder name uses full form
    image_prefix = image_prefix_map.get(dataset_type)  # Image prefix uses mapped form

    # Map the correct abbreviation for the parameter in file names
    parameter_abbr = 'sg' if parameter_type == 'gas_saturation' else 'dP' if parameter_type == 'pressure_buildup' else parameter_type.lower()

    # Construct the folder path based on the dataset type and parameter type
    param_folder = f"samples_{folder_name}_dataset_{parameter_type}"
    img_folder = os.path.join(base_dir, param_folder)
    
    # Print the constructed directory path for verification
    print(f"Looking for images in directory: {img_folder}")
    
    # Check if the folder exists
    if not os.path.exists(img_folder):
        print(f"Directory {img_folder} does not exist.")
        return
    
    # Gather image files in order of time frames, matching the naming convention
    img_files = [
        os.path.join(img_folder, f"{image_prefix}_{parameter_abbr}_sample_{sample_index + 1}_time_{time_label}.png")
        for time_label in time_labels
    ]
    
    # Verify each file's existence and print any missing files
    existing_files = []
    missing_files = []
    for img in img_files:
        if os.path.exists(img):
            existing_files.append(img)
        else:
            missing_files.append(img)
    
    # Print information about missing files
    if missing_files:
        print("The following files were not found:")
        for missing_file in missing_files:
            print(missing_file)
    
    # Proceed only if there are images to create the GIF
    if existing_files:
        images = [imageio.imread(img) for img in existing_files]
        gif_dir = os.path.join(base_dir, 
                               #"GIFs_gas_saturation", # if you create GIF for gas saturation
                               "GIFs_pressure_buildup" # if you create GIF for pressure buildup 
        )
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"{dataset_type}_{parameter_type}_sample_{sample_index + 1}.gif")
        imageio.mimsave(gif_path, images, duration=duration / 1000)  # Duration in seconds
        print(f"GIF saved for {parameter_type} in {gif_path}")
    else:
        print(f"No images found for {parameter_type} Sample {sample_index + 1}")
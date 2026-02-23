##################################################################
# 2025 - King Abdullah University of Science and Technology (KAUST)
#
# Author: Vittoria De Pellegrini 
#
# Description: utility script. This script includes helper functions.
# 
##################################################################

# Import libraries 
import torch
import random
import numpy as np

# Set fixed random seed for reproducibility in PyTorch
def set_seed(seed):
    """Sets all random seeds to a fixed value and takes out any
    randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed number

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True

# Define function to adjust pressure buildup data for correct plotting
def dnorm_dP(dP):
    """
    Adjusts pressure buildup data to ensure it is plotted correctly.

    This function applies a scaling factor and offset to the input pressure 
    buildup data so that the values are suitable for accurate visualization.

    Parameters
    ----------
    dP : float or ndarray
        Input pressure buildup values to be adjusted.

    Returns
    -------
    float or ndarray
        Adjusted pressure buildup values for plotting.

    Notes
    -----
    The adjustment formula is:
        dP = dP * scale + offset
    where:
        scale = 18.772821433027488
        offset = 4.172939172019009
    """
    dP = dP * 18.772821433027488
    dP = dP + 4.172939172019009 
    return dP 

# Define denormalization function for co2 gas saturation (training dataset) -> Global Normalization 
def denormalize_co2_train_global(co2_data, min_train_sg, max_train_sg):
    """
    Denormalizes CO₂ gas saturation data for the training dataset.

    Parameters:
    ----------
    co2_data : torch.Tensor or np.ndarray
        Normalized CO₂ gas saturation data in the range [-1, 1].
    min_train_sg : float
        Minimum value of CO₂ gas saturation in the training dataset.
    max_train_sg : float
        Maximum value of CO₂ gas saturation in the training dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized CO₂ gas saturation data in the range [min_train_sg, max_train_sg].
    """
    return (co2_data + 1) / 2 * (max_train_sg - min_train_sg) + min_train_sg

# Define denormalization function for pressure buildup (training dataset) -> Global Normalization
def denormalize_dP_train_global(pressure_data, min_train_dP, max_train_dP):
    """
    Denormalizes pressure data for the training dataset.

    Parameters:
    ----------
    pressure_data : torch.Tensor or np.ndarray
        Normalized pressure data in the range [-1, 1].
    min_train_dP : float
        Minimum value of pressure in the training dataset.
    max_train_dP : float
        Maximum value of pressure in the training dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized pressure data in the range [min_train_dP, max_train_dP].
    """
    return (pressure_data + 1) / 2 * (max_train_dP - min_train_dP) + min_train_dP 

# Define denormalization function for co2 gas saturation (training dataset) -> Local Normalization
def denormalize_co2_train(co2_data, min_train_sg, max_train_sg):
    return (co2_data + 1) / 2 * (max_train_sg[:, None] - min_train_sg[:, None]) + min_train_sg[:, None]

# Define denormalization function for pressure buildup (training dataset) -> Local Normalization
def denormalize_dP_train(pressure_data, min_train_dP, max_train_dP):
    return (pressure_data + 1) / 2 * (max_train_dP[:, None] - min_train_dP[:, None]) + min_train_dP[:, None]

# Define denormalization function for porosity map (training dataset) -> Global Normalization
def denormalize_por_train(por_data, min_train_por, max_train_por):
    """
    Denormalizes porosity data for the training dataset.

    Parameters:
    ----------
    por_data : torch.Tensor or np.ndarray
        Normalized porosity data in the range [-1, 1].
    min_train_por : float
        Minimum value of porosity in the training dataset.
    max_train_por : float
        Maximum value of porosity in the training dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized porosity data in the range [min_train_por, max_train_por].
    """
    return (por_data + 1) / 2 * (max_train_por - min_train_por) + min_train_por  

# Define denormalization function for co2 gas saturation (validation dataset) -> Global Normalization
def denormalize_co2_val_global(co2_data, min_val_sg, max_val_sg):
    """
    Denormalizes CO₂ gas saturation data for the validation dataset.

    Parameters:
    ----------
    co2_data : torch.Tensor or np.ndarray
        Normalized CO₂ gas saturation data in the range [-1, 1].
    min_val_sg : float
        Minimum value of CO₂ gas saturation in the validation dataset.
    max_val_sg : float
        Maximum value of CO₂ gas saturation in the validation dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized CO₂ gas saturation data in the range [min_val_sg, max_val_sg].
    """
    return (co2_data + 1) / 2 * (max_val_sg - min_val_sg) + min_val_sg  

# Define denormalization function for pressure buildup (validation dataset) -> Global Normalization
def denormalize_dP_val_global(pressure_data, min_val_dP, max_val_dP):
    """
    Denormalizes pressure data for the validation dataset.

    Parameters:
    ----------
    pressure_data : torch.Tensor or np.ndarray
        Normalized pressure data in the range [-1, 1].
    min_val_dP : float
        Minimum value of pressure in the validation dataset.
    max_val_dP : float
        Maximum value of pressure in the validation dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized pressure data in the range [min_val_dP, max_val_dP].
    """
    return (pressure_data + 1) / 2 * (max_val_dP - min_val_dP) + min_val_dP  

# Define denormalization function for co2 gas saturation (validation dataset) -> Local Normalization
def denormalize_co2_val(co2_data, min_val_sg, max_val_sg):
    return (co2_data + 1) / 2 * (max_val_sg[:, None] - min_val_sg[:, None]) + min_val_sg[:, None]

# Define denormalization function for pressure buildup (validation dataset) -> Local Normalization
def denormalize_dP_val(pressure_data, min_val_dP, max_val_dP):
    return (pressure_data + 1) / 2 * (max_val_dP[:, None] - min_val_dP[:, None]) + min_val_dP[:, None]

# Define denormalization function for porosity map (validation dataset) -> Global Normalization
def denormalize_por_val(por_data, min_val_por, max_val_por):
    """
    Denormalizes porosity data for the validation dataset.

    Parameters:
    ----------
    por_data : torch.Tensor or np.ndarray
        Normalized porosity data in the range [-1, 1].
    min_val_por : float
        Minimum value of porosity in the validation dataset.
    max_val_por : float
        Maximum value of porosity in the validation dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized porosity data in the range [min_val_por, max_val_por].
    """
    return (por_data + 1) / 2 * (max_val_por - min_val_por) + min_val_por  

# Define denormalization function for co2 gas saturation (test dataset) -> Global Normalization
def denormalize_co2_test_global(co2_data, min_test_sg, max_test_sg):
    """
    Denormalizes CO₂ gas saturation data for the test dataset.

    Parameters:
    ----------
    co2_data : torch.Tensor or np.ndarray
        Normalized CO₂ gas saturation data in the range [-1, 1].
    min_test_sg : float
        Minimum value of CO₂ gas saturation in the test dataset.
    max_test_sg : float
        Maximum value of CO₂ gas saturation in the test dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized CO₂ gas saturation data in the range [min_test_sg, max_test_sg].
    """
    return (co2_data + 1) / 2 * (max_test_sg - min_test_sg) + min_test_sg 

# Define denormalization function for pressure buildup (test dataset) -> Global Normalization
def denormalize_dP_test_global(pressure_data, min_test_dP, max_test_dP):
    """
    Denormalizes pressure data for the test dataset.

    Parameters:
    ----------
    pressure_data : torch.Tensor or np.ndarray
        Normalized pressure data in the range [-1, 1].
    min_test_dP : float
        Minimum value of pressure in the test dataset.
    max_test_dP : float
        Maximum value of pressure in the test dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized pressure data in the range [min_test_dP, max_test_dP].
    """
    return (pressure_data + 1) / 2 * (max_test_dP - min_test_dP) + min_test_dP  

# Define denormalization function for co2 gas saturation (test dataset) -> Local Normalization
def denormalize_co2_test(co2_data, min_test_sg, max_test_sg):
    return (co2_data + 1) / 2 * (max_test_sg[:, None] - min_test_sg[:, None]) + min_test_sg[:, None]

# Define denormalization function for pressure buildup (test dataset) -> Local Normalization
def denormalize_dP_test(pressure_data, min_test_dP, max_test_dP):
    return (pressure_data + 1) / 2 * (max_test_dP[:, None] - min_test_dP[:, None]) + min_test_dP[:, None]

# Define denormalization function for porosity map (test dataset) -> Global Normalization
def denormalize_por_test(por_data, min_test_por, max_test_por):
    """
    Denormalizes porosity data for the test dataset.

    Parameters:
    ----------
    por_data : torch.Tensor or np.ndarray
        Normalized porosity data in the range [-1, 1].
    min_test_por : float
        Minimum value of porosity in the test dataset.
    max_test_por : float
        Maximum value of porosity in the test dataset.

    Returns:
    -------
    torch.Tensor or np.ndarray
        Denormalized porosity data in the range [min_test_por, max_test_por].
    """
    return (por_data + 1) / 2 * (max_test_por - min_test_por) + min_test_por 

# Define custom R² function
def custom_r2_score(y_true, y_pred):
    """
    Custom implementation of R² score excluding zero values in the ground truth.
    """
    nonzero_mask = y_true != 0
    y_true_nonzero = y_true[nonzero_mask]
    y_pred_nonzero = y_pred[nonzero_mask]

    y_mean = np.mean(y_true_nonzero)
    ss_res = np.sum((y_true_nonzero - y_pred_nonzero) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true_nonzero - y_mean) ** 2)  # Total sum of squares

    if ss_tot == 0:  # Handle edge case where all y_true_nonzero values are the same
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)
![LOGO](https://github.com/DeepWave-KAUST/DMFCO2-dev/blob/main/asset/DMFCO2_Banner_Repo.png)

Reproducible material for **A Conditional Diffusion Model for CO₂ Monitoring and Forecasting in Heterogeneous Geological Formations - De Pellegrini V., Wamriew D., Alkhalifah T.**, and **Towards generative modeling of CO2 geological storage with latent conditional diffusion models - De Pellegrini V., Alkhalifah T.**

## :bar_chart: Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo.
* :open_file_folder: **co2_dataset**: folder containing data (or instructions on how to retrieve the data).
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details).
* :open_file_folder: **dmfco2**: python library containing routines for the notebook.
* :open_file_folder: **results**: folder containing training-validation-testing results for VQ-VAE+GAN and Diffusion models (or instructions on how to access the results).
* :open_file_folder: **trained_models**: folder containing VQ-VAE+GAN and Diffusion trained models (or instructions on how to retrieve the trained models).

## :fire: Notebooks
The following notebooks are provided:

- :green_book: ``01_Dataset_Samples_Visualization.ipynb``: notebook performing analysis and visualization of the open-source CO₂ dataset.
- :blue_book: ``02_Training_VQ-VAE+GAN_CO2_Gas_Saturation.ipynb``: notebook performing training of the VQ-VAE + GAN model for CO₂ Gas Saturation variable.
- :blue_book: ``02_Training_VQ-VAE+GAN_Pressure_BuildUp.ipynb``: notebook performing training of the VQ-VAE + GAN model for Pressure Build-Up variable.
- :blue_book: ``02_Validating_VQ-VAE+GAN_CO2_Gas_Saturation.ipynb``: notebook performing validation of the VQ-VAE + GAN model for CO₂ Gas Saturation variable.
- :blue_book: ``02_Validating_VQ-VAE+GAN_Pressure_BuildUp.ipynb``: notebook performing validation of the VQ-VAE + GAN model for Pressure Build-Up variable.
- :blue_book: ``02_Testing_VQ-VAE+GAN_CO2_Gas_Saturation.ipynb``: notebook performing testing of the VQ-VAE + GAN model for CO₂ Gas Saturation variable.
- :blue_book: ``02_Testing_VQ-VAE+GAN_Pressure_BuildUp.ipynb``: notebook performing testing of the VQ-VAE + GAN model for Pressure Build-Up variable.
- :orange_book: ``03_Training_Latent_Conditional_Diffusion_CO2_Gas_Saturation_multiple_conditions.ipynb``: notebook performing training of the latent conditional diffusion model for CO₂ Gas Saturation variable.
- :orange_book: ``03_Training_Latent_Conditional_Diffusion_Pressure_BuildUp_multiple_conditions.ipynb``: notebook performing training of the latent conditional diffusion model for Pressure Build-Up variable.
- :orange_book: ``03_Validating_Latent_Conditional_Diffusion_CO2_Gas_Saturation_multiple_conditions.ipynb``: notebook performing validation of the latent conditional diffusion model for CO₂ Gas Saturation variable.
- :orange_book: ``03_Validating_Latent_Conditional_Diffusion_Pressure_BuildUp_multiple_conditions.ipynb``: notebook performing validation of the latent conditional diffusion model for Pressure Build-Up variable.
- :orange_book: ``03_Testing_Latent_Conditional_Diffusion_CO2_Gas_Saturation_multiple_conditions.ipynb``: notebook performing testing of the latent conditional diffusion model for CO₂ Gas Saturation variable.
- :orange_book: ``03_Testing_Latent_Conditional_Diffusion_Pressure_BuildUp_multiple_conditions.ipynb``: notebook performing testing of the latent conditional diffusion model for Pressure Build-Up variable.

## CO₂ Dataset Info (co2_dataset)
To ensure reproducibility, we provide the **official link** for the CO₂ Geological Sequestration Open Source Dataset. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing).

**1. Training Dataset (n_samples = 4,500)**

- **CO₂ Gas Saturation**:
  - **Inputs/Targets**: `sg_train_a.pt -> sg_train_inputs.pt`
  - **Outputs/Conditions**: `sg_train_u.pt -> sg_train_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/training_dataset_gas_saturation/sg_train_inputs.pt`
    - `/.../co2_dataset/training_dataset_gas_saturation/sg_train_outputs.pt`

- **Pressure Build-Up**:
  - **Inputs/Targets**: `dP_train_a.pt -> dP_train_inputs.pt`
  - **Outputs/Conditions**: `dP_train_u.pt -> dP_train_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/training_dataset_pressure_buildup/dP_train_inputs.pt`
    - `/.../co2_dataset/training_dataset_pressure_buildup/dP_train_outputs.pt`

**2. Validation Dataset (n_samples = 500)**

- **CO₂ Gas Saturation**:
  - **Inputs/Targets**: `sg_val_a.pt -> sg_val_inputs.pt`
  - **Outputs/Conditions**: `sg_val_u.pt -> sg_val_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/validation_dataset_gas_saturation/sg_val_inputs.pt`
    - `/.../co2_dataset/validation_dataset_gas_saturation/sg_val_outputs.pt`

- **Pressure Build-Up**:
  - **Inputs/Targets**: `dP_val_a.pt -> dP_val_inputs.pt`
  - **Outputs/Conditions**: `dP_val_u.pt -> dP_val_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/validation_dataset_pressure_buildup/dP_val_inputs.pt`
    - `/.../co2_dataset/validation_dataset_pressure_buildup/dP_val_outputs.pt`

**3. Test Dataset (n_samples = 500)**

- **CO₂ Gas Saturation**:
  - **Inputs/Targets**: `sg_test_a.pt -> sg_test_inputs.pt`
  - **Outputs/Conditions**: `sg_test_u.pt -> sg_test_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/test_dataset_gas_saturation/sg_test_inputs.pt`
    - `/.../co2_dataset/test_dataset_gas_saturation/sg_test_outputs.pt`

- **Pressure Build-Up**:
  - **Inputs/Targets**: `dP_test_a.pt -> dP_test_inputs.pt`
  - **Outputs/Conditions**: `dP_test_u.pt -> dP_test_outputs.pt`
  - Respectively, create the following folders:
    - `/.../co2_dataset/test_dataset_pressure_buildup/dP_test_inputs.pt`
    - `/.../co2_dataset/test_dataset_pressure_buildup/dP_test_outputs.pt`

## Pre-Trained VGG-16 Model, Trained VQ-VAE+GAN Models and Trained Latent Conditional Diffusion Models (trained_models)
To ensure reproducibility, we provide the link for the pre-trained models.

* **Pre-Trained VGG-16 Model:**
The pre-trained model (**vgg.pth**) can be downloaded from the official [page](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth).

* **Trained VQ-VAE + GAN Models:**
The trained models (**vqvae_model.pth**) are available for download:
  - CO₂ Gas Saturation Model [here](https://drive.google.com/drive/folders/1tVmd8f_2iwMlMhZ1jMJ0NQUWwU0ZXXlW?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto). Then use `unzip` to extract the content to `/.../trained_models/gas_saturation_models/vqvae/best_model`.
  - Pressure Build-Up Model [here](https://drive.google.com/drive/folders/1tVmd8f_2iwMlMhZ1jMJ0NQUWwU0ZXXlW?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto). Then use `unzip` to extract the content to `/.../trained_models/pressure_buildup_models/vqvae/best_model`.

* **Trained Latent Conditional Diffusion Models:**
The trained models (**diffusion_model.pth**) are available for download:
  - CO₂ Gas Saturation Model [here](https://drive.google.com/drive/folders/1tVmd8f_2iwMlMhZ1jMJ0NQUWwU0ZXXlW?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto). Then use `unzip` to extract the content to `/.../trained_models/gas_saturation_models/diffusion_model/best_model`.
  - Pressure Build-Up Model [here](https://drive.google.com/drive/folders/1tVmd8f_2iwMlMhZ1jMJ0NQUWwU0ZXXlW?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto). Then use `unzip` to extract the content to `/.../trained_models/pressure_buildup_models/diffusion_model/best_model`. 

## Results (results)
This folder contains **TensorBoard log files** and **training curves** generated during the training of the VQ-VAE+GAN and Diffusion Models.  The results are available [here](https://drive.google.com/drive/folders/1CxTbF3JSMsbc0RPzMERxUGe4mS5l4UCy?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto). Then use `unzip` to extract the content to `/.../results`.

## :space_invader: :robot: Getting started 

> [!IMPORTANT]
>To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.
>
>Simply run:
>```
>./install_env.sh
>```
>It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 
>
>Remember to always activate the environment by typing:
>```
>conda activate dmfco2
>```
>
>After that you can simply install your package:
>```
>pip install .
>```
>or in developer mode:
>```
>pip install -e .
>```

> [!NOTE]
>**Disclaimer:**  All experiments have been carried on a Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment configurations may be required for different combinations of workstation and GPU.

## :alien: :flying_saucer: :cow2: Cite us 

```bibtex
@article{eage:/content/papers/10.3997/2214-4609.202510212,
   author = "Pellegrini, V. De and Wamriew, D. and Alkhalifah, T.",
   title = "A Conditional Diffusion Model for CO₂ Monitoring and Forecasting in Heterogeneous Geological Formations", 
   journal= "",
   year = "2025",
   volume = "2025",
   number = "1",
   pages = "1-5",
   doi = "https://doi.org/10.3997/2214-4609.202510212",
   url = "https://www.earthdoc.org/content/papers/10.3997/2214-4609.202510212",
   publisher = "European Association of Geoscientists &amp; Engineers",
   issn = "2214-4609",
   type = "",
  }
@inproceedings{de2025towards,
  title={Towards generative modeling of CO2 geological storage with latent conditional diffusion models},
  author={De Pellegrini, Vittoria and Alkhalifah, Tariq},
  booktitle={International Meeting for Applied Geoscience \& Energy Expanded Abstracts},
  volume={44},
  number={1},
  pages={1290--1294},
  organization={Society of Exploration Geophysicists}
}
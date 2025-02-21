# AstroM<sup>3</sup>: A self-supervised multimodal model for astronomy

![Model Overview](images/astroclip.png)
*Figure 1: Overview of the multimodal CLIP framework adapted for astronomy, incorporating three data modalities: photometric time-series, spectra, and metadata. Each modality is processed by a dedicated encoder to create embeddings, which are then mapped into a shared embedding space through projection heads. Pairwise similarity matrices align the embeddings across modalities, and a symmetric cross-entropy loss, computed over these matrices, optimizes the model. The total loss, derived from all pairwise losses, guides the model’s trimodal learning.*


## Requirements

Before running the scripts, you need to install the required dependencies:
   ```sh
   mamba env create -f environment.yml
   ```

Set up WandB for experiment tracking:
   ```sh
   wandb login
   ```

Set up MySQL database if using Optuna's storage for hyperparameter tuning.


## Downloading the Data

## This Repo Structure:
This repository includes the following folders:
- `core/`: Contains all the essential code.
- `dev/`: Holds outdated code that may not work but could serve as a useful reference.
- `models/`: Contains code for key Informer blocks.
- `notebooks/`: Contains various notebooks, though some may no longer function—use for reference only.
- `util/`: Includes an early stopping class and utility functions for parallel zip processing and data handling.

## Project Structure


The `core/` folder is organized into several key files and modules, each responsible for a different part of the workflow.

- **`dataset.py`**: Defines `PSMDataset`, a custom dataset class that handles data loading, filtering, and preprocessing for photometry, spectra, and metadata. 

- **`loss.py`**: Contains `CLIPLoss`, a custom loss function designed for multimodal alignment to align representations from different data types.

- **`main.py`**: The main script to train and evaluate the models. It sets up the configuration, loads data, initializes the model, and manages the training process.

- **`model.py`**: Defines the model architectures for each modality:
  - `Informer` for photometric data,
  - `GalSpecNet` for spectroscopic data,
  - `MetaModel` for metadata, and
  - `AstroM3` as the combined multimodal model.

- **`trainer.py`**: Manages the training and evaluation processes, including logging, early stopping, and metric tracking.

- **`tune.py`**: Script for hyperparameter tuning using Optuna. This script explores different hyperparameter configurations to optimize model performance.

## Configuration Setup

The function `get_config()` in `main.py` provides a structured configuration for training and evaluating the model. It defines various parameters that control data paths, model architecture, and training behaviors. Adjusting these parameters allows for fine-tuning the model and adapting it to specific tasks. The configuration returned by `get_config()` is used throughout the training process to ensure consistency. If you want to change any parameter it should be changed here.

### How to Use `get_config()`

1. **Basic Structure**: The configuration dictionary includes essential parameters, divided into sections for project settings, data loading, model architecture, and training hyperparameters.

2. **Modifying Parameters**:
   - Open `main.py` and locate the `get_config()` function.
   - Modify the fields to suit your requirements.

3. **Key Parameters**:
   - **Project Settings**:
     - `'project'`: Name of the project in WandB (e.g., `'AstroCLIPResults3'`).
     - `'mode'`: Specifies which modality is used (e.g., `'spectra'` for spectra classification, `'photo` for photometry classification, `'meta'` for metadata classification, `'clip'` for CLIP pre-training using all three modalities, `'all'` for classification using photometry, spectra and metadata).
     - `'random_seed'`: Sets a fixed random seed for reproducibility.
     - `'use_wandb'`: Enable or disable WandB for tracking (True/False).
     - `'save_weights'`: Enable or disable saving weights
     - `'use_pretrain'`: If you want to use pre-trained model, specify the path to the weights here

   - **Data Settings**:
     - `'data_root'`: Directory path to your data.
     - `'file'`: Base name of the file used for loading datasets.
     - `'classes'`: List of class labels in your dataset which you want to use.
     - `'meta_cols'`, `'photo_cols'`: Columns for metadata and photometry features.
     - `'min_samples'`, `'max_samples'`: Set minimum and maximum sample count per class.

   - **Model Architecture**:
     - **Photometry Model**:
       - `'p_enc_in'`, `'p_d_model'`, `'p_dropout'`, `'p_n_heads'`: Parameters for the photometry model architecture.
     - **Spectra Model**:
       - `'s_conv_channels'`, `'s_dropout'`: Convolutional channels and dropout settings.
     - **Metadata Model**:
       - `'m_hidden_dim'`, `'m_dropout'`: Hidden layer dimension and dropout settings for metadata.
     - **Multimodal Fusion**:
       - `'fusion'`: Specifies the fusion strategy (e.g., `'avg'`, `'concat'`).

   - **Training Hyperparameters**:
     - `'batch_size'`, `'lr'`, `'epochs'`: Basic training settings.
     - `'scheduler'`: Learning rate scheduler type (e.g., `'ExponentialLR'`, `'ReduceLROnPlateau'`).
     - `'early_stopping_patience'`: Epochs to wait before early stopping if validation loss doesn’t improve.

### Using Previous Configuration as a Template

If you want to copy parameters from a previous WandB run, you can set the `'config_from'` field to the specific WandB run path:

```
'config_from': 'username/projectname/run_id'
```

This configuration setup allows you to copy specific parameters from a previous WandB run and overwrite them in the current configuration. Only the parameters listed below will be copied:
   - Dropout settings:
     - `p_dropout`: Photometry model dropout rate
     - `s_dropout`: Spectra model dropout rate
     - `m_dropout`: Metadata model dropout rate
   - Learning and optimization settings:
     - `lr`: Learning rate
     - `beta1`: Beta1 for Adam optimizer
     - `weight_decay`: Weight decay for optimizer
     - `epochs`: Total number of training epochs
   - Early stopping and scheduler:
     - `early_stopping_patience`: Patience for early stopping
     - `factor`: Factor by which the learning rate is reduced
     - `patience`: Patience for learning rate scheduler
     - `warmup`: Enable warmup
     - `warmup_epochs`: Number of warmup epochs
   - Gradient clipping:
     - `clip_grad`: Enable gradient clipping
     - `clip_value`: Clip value for gradients
   - Pre-training and data handling:
     - `use_pretrain`: Path to pre-trained weights
     - `freeze`: Whether to freeze model layers during training
     - `phased`: Flag to enable phased data processing
     - `p_aux`, `s_aux`, `s_err`: Auxiliary flags for data processing
     - `file`: Path to the dataset file

It's a very useful feature, especially when you're doing a lot of tuning, as these are the parameters that frequently change. It also significantly improves reproducibility.

## Training 

After setting all the configuration, to train the model simply run:
```sh
python main.py
```

[Optional] You might need to modify python path:
```sh
export PYTHONPATH=$PYTHONPATH:/<root_dir>/
```
instead of `<root_dir>` use the full path to this repo.

## Tuning

For hyperparameter optimization using Optuna:
```sh
python tune.py
```
The `tune.py` script sets up an Optuna study and performs hyperparameter tuning based on the specified search space. But in a nutshell it does the same thing as `main.py`.

## Citation
If you find this repo useful, please cite our paper.
```
@inproceedings{rizhko2024self,
  title={Self-supervised Multimodal Model for Astronomy},
  author={Rizhko, Mariia and Bloom, Joshua S},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```

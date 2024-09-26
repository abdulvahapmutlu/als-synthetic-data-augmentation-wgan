# ALS Synthetic Data Augmentation Using WGAN

## Introduction

This project aims to address the lack of EEG signals for ALS (Amyotrophic Lateral Sclerosis) patients. ALS is a progressive neurodegenerative disease, and there is limited availability of data to train machine learning models for its diagnosis and study. To overcome this, a Wasserstein GAN (WGAN) model has been employed to generate synthetic EEG signals that can augment the dataset for ALS patients. The synthetic data produced by this model aims to improve the performance of diagnostic algorithms by providing more training samples.

## Project Overview

This repository contains the implementation of a WGAN to generate synthetic EEG signals for ALS patients. The project is divided into several components, including data preprocessing, model training, and synthetic signal generation.

The repository is structured as follows:

- **models/**: Saved trained models, including the generator and critic, in Keras format.
- **scripts/**: Python scripts for data preprocessing, training the WGAN model, and generating synthetic signals.
- **results/**: Contains output plots showing the loss curves of the generator and critic during the training phase.

## Installation and Setup

### 1. Clone the Repository

To get started, clone this repository to your local machine:

```
git clone https://github.com/abdulvahapmutlu/als-synthetic-data-augmentation-wgan.git
```

### 2. Install Dependencies
To ensure that the code runs smoothly, you need to install the required Python packages. These dependencies are listed in the requirements.txt file. Install them by running:
```
pip install -r requirements.txt
```
The main packages used in this project are:

Keras for building and training the WGAN model.
NumPy for efficient numerical operations.
SciPy for handling .mat files and saving/loading EEG data.
Matplotlib for plotting and visualizing the training loss curves.

### 3. Dataset Preparation
Place your dataset of EEG signals in the data/ALS/ directory. The structure of the data folder should be as follows:

data/ALS/0/: This folder should contain the .mat files for non-ALS patients, with filenames starting with 0_.
data/ALS/1/: This folder should contain the .mat files for ALS patients, with filenames starting with 1_.
Each .mat file is expected to contain the EEG signals in the form of a numeric matrix, which the preprocessing script will normalize and load for training.

## Running the Project
### 1. Data Preprocessing
Before training the WGAN, the EEG signals need to be preprocessed. This involves loading the signals from .mat files, normalizing them between -1 and 1, and stacking them into arrays for training.

To run the data preprocessing, use the following command:

```
python scripts/data_preprocessing.py
```
This script will:

Load EEG signals from the provided dataset.
Normalize the signals.
Output the total number of samples loaded and their respective labels.

### 2. Training the WGAN
Once the data is preprocessed, you can train the WGAN model. The model consists of a generator that synthesizes new EEG signals and a critic (discriminator) that evaluates their authenticity.

To train the WGAN, use the following command:
```
python scripts/wgan_training.py
```
During training, the generator and critic models will be updated iteratively. The critic model is updated multiple times per generator update, as per the WGAN architecture. Loss curves for both models will be plotted and saved in the results/ folder.

The training script includes the following hyperparameters:

epochs: The number of training epochs (set to 300 by default).
batch_size: The number of samples processed in each batch.
n_critic: The number of critic updates per generator update (set to 5).
clip_value: The value for clipping critic model weights (set to 0.01).
At the end of training, the generator and critic models will be saved in the models/ folder as .keras files.

### 3. Generating New Signals
After training, you can generate synthetic EEG signals using the trained generator. To do this, run the following command:

```
python scripts/signal_generation.py
```
This script will generate 10,000 .mat files, each containing 2560 synthetic EEG signals with a signal length of 32. The files will be saved in the generated_files/ directory.

### 4. Viewing Results
The plot showing the discriminator and generator loss curves over the training epochs will be saved in the results/ folder. This can help you evaluate how the models performed during training.

### 5. Model Saving and Loading
The trained generator and critic models are saved after training. If you want to load these models for further use, you can do so using Keras' load_model function. For example, to load the generator:
```
from keras.models import load_model
```

## Load the generator model
generator = load_model('models/wgan_generator.keras', custom_objects={'wasserstein_loss': wasserstein_loss})

## Future Work
This project opens several opportunities for further development:

Model Optimization: Experimenting with different architectures and hyperparameters to improve the quality of generated signals.
Data Augmentation: Expanding the dataset with additional synthetic samples and analyzing their impact on ALS diagnosis models.
Comparison with Other GAN Variants: Comparing the performance of WGAN with other GAN models (e.g., DCGAN, StyleGAN) to assess the most effective technique for EEG signal generation.

## License
This project is licensed under the Apache 2.0 License.

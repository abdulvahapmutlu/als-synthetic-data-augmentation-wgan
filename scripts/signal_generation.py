from keras.models import load_model
import os
import numpy as np
import keras.backend as K
from scipy.io import savemat  # Import savemat to save files in .mat format

# WGAN loss function
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Load the trained generator model
generator = load_model('wgan_generator.keras', custom_objects={'wasserstein_loss': wasserstein_loss})

# Function to generate a batch of new signals with the label "0" only
def generate_new_signals(generator, latent_dim, num_samples, label):
    # Generate random noise
    noise = np.random.randn(num_samples, latent_dim)
    # Create an array of labels with "0"
    labels = np.zeros(num_samples, dtype=int)
    
    # Generate signals using the generator model
    generated_signals = generator.predict([noise, labels])
    
    return generated_signals

# Parameters for signal generation
num_samples = 2560  # Number of samples per file
latent_dim = 100    # Dimensionality of the latent space
num_classes = 2     # Number of classes (generator was trained with 2)

# Output directory for generated files
output_dir = "generated_files"
os.makedirs(output_dir, exist_ok=True)

# Generate and save 10,000 files
num_files = 10000
for i in range(num_files):
    # Generate signals with label "0" only
    generated_signals = generate_new_signals(generator, latent_dim, num_samples, label=0)
    
    # Ensure the shape is (2560, 32)
    if generated_signals.shape[1] != 32:
        raise ValueError("Generated signals do not have the expected shape (2560, 32). Check the generator architecture.")
    
    # Save the generated signals to a .mat file
    output_path = os.path.join(output_dir, f"generated_signals_{i + 1}.mat")
    savemat(output_path, {'generated_signals': generated_signals})
    
    if (i + 1) % 100 == 0:  # Print progress every 100 files
        print(f"Saved {i + 1}/{num_files} files.")

print("All files have been generated and saved successfully.")

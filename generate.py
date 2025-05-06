import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import the VAE model definition
# We will load LATENT_DIM from the saved hyperparameters
from model import VAE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH # Removed LATENT_DIM import

# --- Configuration ---
MODEL_PATH = './best_vae_stroke_model (1).pth' # Path to the saved dictionary

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Instantiate and Load the Trained VAE Model ---
# Load the saved dictionary containing state_dict and hyperparameters
if os.path.exists(MODEL_PATH):
    print(f"Loading model weights and hyperparameters from {MODEL_PATH}")
    # Load the entire dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Extract the hyperparameters and model state dictionary
    loaded_hyperparameters = checkpoint['hyperparameters']
    loaded_model_state_dict = checkpoint['model_state_dict']
    loaded_best_val_f1 = checkpoint['best_val_f1'] # Optional: print this out

    # Get the latent_dim from the loaded hyperparameters
    loaded_latent_dim = loaded_hyperparameters['latent_dim']

    # Instantiate the VAE model with the correct latent dimension
    model = VAE(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, loaded_latent_dim).to(device)

    # Load the state dictionary into the instantiated model
    model.load_state_dict(loaded_model_state_dict)
    model.eval() # Set model to evaluation mode

    print("Model loaded successfully.")
    print(f"Model configured with Latent Dimension: {loaded_latent_dim}")
    print(f"Best Validation F1 from training: {loaded_best_val_f1:.4f}")
    print(f"Full Hyperparameters: {loaded_hyperparameters}")

else:
    print(f"Error: Model weights and hyperparameters not found at {MODEL_PATH}")
    print("Please run train.py first to train and save the model.")
    exit() # Exit if model not found


# --- Generate an Image ---
print("\nGenerating an image by sampling from the latent space...")

# Sample a single random latent vector from the standard normal distribution (the prior)
# Use the loaded_latent_dim for the random vector size
random_latent_vector = torch.randn(1, loaded_latent_dim).to(device)

# Pass the latent vector through the decoder part of the VAE
with torch.no_grad(): # No need to calculate gradients for generation
    # 1. Pass through the initial dense layer in the decoder
    decoder_input = model.decoder_input(random_latent_vector)

    # 2. Reshape the output to the spatial dimensions expected by the ConvTranspose layers
    # Use the dynamic calculation from the model's decoder_input layer size
    # The expected channels before reshape are determined by decoder_input.out_features / (4*4)
    decoder_input = decoder_input.view(-1, model.decoder_input.out_features // (4*4), 4, 4)

    # 3. Pass through the transposed convolutional layers to generate the image
    generated_image_tensor = model.decoder(decoder_input) # Output shape: [1, 1, IMG_HEIGHT, IMG_WIDTH]

# --- Visualize the Generated Image ---
print("Displaying the generated image...")

# Move the tensor to CPU and convert to a NumPy array
# .squeeze() removes dimensions of size 1 (like the batch and channel dimensions)
generated_image_np = generated_image_tensor.squeeze().cpu().numpy()

# Display the image using Matplotlib
plt.imshow(generated_image_np, cmap='gray') # Use 'gray' colormap for grayscale images
plt.title("Generated Image from Latent Space")
plt.axis('off') # Hide axes
plt.show()

print("Image generation and display complete.")
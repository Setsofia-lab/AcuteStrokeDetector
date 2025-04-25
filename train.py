import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Import the VAE model definition from model.py
from model import VAE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, LATENT_DIM

# Import data loading components from data_prep.py
from preprocess import train_loader, val_loader, class_weights_tensor # Assuming test_loader is not needed during training

# --- Training Configuration ---
NUM_EPOCHS = 50 # Number of times to loop through the entire training dataset
LEARNING_RATE = 1e-3 # Step size for the optimizer
BETA = 1.0 # Weight for the KL divergence term in the VAE loss (often set to 1.0)
MODEL_SAVE_PATH = './vae_stroke_model.pth' # Path to save the best model weights

# --- Device Configuration ---
# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Instantiate the VAE Model ---
model = VAE(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, LATENT_DIM).to(device)
print("VAE Model Architecture:")
print(model)

# --- Optimizer ---
# Adam is a popular optimizer that adjusts the learning rate for each parameter
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- VAE Loss Function ---
# The VAE loss consists of two parts: Reconstruction Loss and KL Divergence Loss
def vae_loss_function(reconstructed_x, x, mu, logvar, class_labels, class_weights):
    # 1. Reconstruction Loss (Binary Cross-Entropy)
    # We use BCE because the output of the decoder is pixel values between 0 and 1 (due to Sigmoid)
    # and the input images were normalized to 0-1.
    # We flatten the images to calculate BCE per pixel across the batch.
    # Input x and reconstructed_x are of shape [batch_size, 1, H, W]
    # We need to flatten them to [batch_size, 1*H*W]
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_x.view(-1, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH),
        x.view(-1, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH),
        reduction='none' # Calculate loss per pixel per image
    )
    # The shape of reconstruction_loss is now [batch_size, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH]

    # Apply class weights to the reconstruction loss
    # We need to get the weight for each image in the batch based on its class label
    # class_labels is of shape [batch_size]
    # class_weights is a tensor [weight_for_class_0, weight_for_class_1]
    # We can use advanced indexing to get the weight for each label
    weights_for_batch = class_weights[class_labels] # Shape: [batch_size]

    # Multiply the per-pixel loss for each image by its corresponding class weight
    # We need to reshape weights_for_batch to match the shape of reconstruction_loss for broadcasting
    weighted_reconstruction_loss = reconstruction_loss * weights_for_batch.unsqueeze(1) # Shape: [batch_size, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH]

    # Sum the weighted reconstruction loss over all pixels for each image, then average over the batch
    weighted_reconstruction_loss = torch.mean(torch.sum(weighted_reconstruction_loss, dim=1)) # Shape: scalar

    # 2. KL Divergence Loss (Regularization)
    # This term encourages the encoder's distribution q(z|x) to be close to the prior p(z) (standard normal).
    # The formula for KL divergence between N(mu, sigma^2) and N(0, 1) is:
    # 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    # We use logvar (log(sigma^2)) directly from the encoder output.
    # KL_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    # Average the KL divergence over the batch
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    kl_divergence /= x.size(0) # Divide by batch size to get the average KL divergence per image

    # 3. Total VAE Loss
    # The total loss is the sum of the weighted reconstruction loss and the KL divergence loss
    # We can weight the KL term with BETA if needed, but BETA=1.0 is common.
    total_loss = weighted_reconstruction_loss + BETA * kl_divergence

    return total_loss, weighted_reconstruction_loss, kl_divergence

# --- Training Loop ---
def train(epoch):
    model.train() # Set model to training mode
    train_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0

    # Iterate over batches in the training data loader
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move images and labels to the appropriate device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device) # Labels are needed for weighted loss

        # Forward pass: Get reconstructed images, mean, and log-variance
        reconstructed_images, mu, logvar = model(images)

        # Calculate the VAE loss
        loss, recon_loss, kl_loss = vae_loss_function(
            reconstructed_images, images, mu, logvar, labels, class_weights_tensor.to(device) # Pass weights to device
        )

        # Backward pass: Calculate gradients
        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Calculate gradients of the loss with respect to model parameters

        # Optimizer step: Update model weights
        optimizer.step()

        # Accumulate loss for reporting
        train_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

        # Print training progress
        if (batch_idx + 1) % 100 == 0: # Print every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(images):.6f}') # Average loss per image in batch

    # Calculate average loss for the epoch
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = recon_loss_sum / len(train_loader.dataset)
    avg_kl_loss = kl_loss_sum / len(train_loader.dataset)

    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')
    print(f'      Average train reconstruction loss: {avg_recon_loss:.4f}')
    print(f'      Average train KL divergence loss: {avg_kl_loss:.4f}')

    return avg_train_loss

# --- Validation Loop ---
def validate(epoch):
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm tracking, etc.)
    val_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0

    # Disable gradient calculation for validation (saves memory and computation)
    with torch.no_grad():
        # Iterate over batches in the validation data loader
        for images, labels in val_loader:
            # Move images and labels to the appropriate device
            images = images.to(device)
            labels = labels.to(device) # Labels are needed for weighted loss

            # Forward pass
            reconstructed_images, mu, logvar = model(images)

            # Calculate the VAE loss
            loss, recon_loss, kl_loss = vae_loss_function(
                reconstructed_images, images, mu, logvar, labels, class_weights_tensor.to(device) # Pass weights to device
            )

            # Accumulate loss
            val_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()

    # Calculate average loss for the epoch
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon_loss = recon_loss_sum / len(val_loader.dataset)
    avg_val_kl_loss = kl_loss_sum / len(val_loader.dataset)

    print(f'====> Epoch: {epoch} Average validation loss: {avg_val_loss:.4f}')
    print(f'      Average validation reconstruction loss: {avg_val_recon_loss:.4f}')
    print(f'      Average validation KL divergence loss: {avg_val_kl_loss:.4f}')

    return avg_val_loss

# --- Main Training Execution ---
if __name__ == '__main__':
    best_val_loss = float('inf') # Initialize with a very high value

    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)

        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved. Saving model to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nTraining finished.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Model weights saved to {MODEL_SAVE_PATH}")


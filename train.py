import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import os
import itertools
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt # Import matplotlib for plotting images

# Import the VAE model definition from model.py
from model import VAE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH # Removed LATENT_DIM as it will be tuned

# Import data loading components from data_prep.py
from preprocess import train_loader, val_loader, class_weights_tensor

# --- Grid Search Hyperparameter Configuration ---
# Define the grid of hyperparameters to search
PARAM_GRID = {
    'learning_rate': [1e-4, 5e-5], #[1e-4,1e-5, 5e-5], # Experiment with learning rates - Adjusted for potentially longer training
    'latent_dim': [128, 256],       # Experiment with latent space sizes - Adjusted for potentially longer training
    'beta': [0.5, 1.0, 2.0]             # Experiment with KL divergence weight - Added a higher value
}

# --- Training Configuration (per hyperparameter combination) ---
NUM_EPOCHS_PER_COMBINATION = 300 # Increased number of epochs
MAX_GRAD_NORM = 1.0             # Maximum norm for gradients
LOG_INTERVAL = 100              # Log training loss every n batches
SAVE_IMAGE_INTERVAL = 50        # Save sample images every n epochs

# --- Evaluation Metric for Hyperparameter Selection ---
def calculate_reconstruction_error(original_images, reconstructed_images):
    """
    Calculates the Binary Cross-Entropy reconstruction error for each image in a batch.
    Assumes batch size is 1 for single image prediction, or returns a tensor for batch.

    Args:
        original_images (torch.Tensor): The original images (shape [batch_size, 1, H, W]).
        reconstructed_images (torch.Tensor): The reconstructed images (shape [batch_size, 1, H, W]).

    Returns:
        torch.Tensor: A tensor of shape [batch_size] containing the BCE error for each image.
    """
    epsilon = 1e-6 # Added epsilon for numerical stability
    reconstructed_images = torch.clamp(reconstructed_images, epsilon, 1 - epsilon)
    original_images = torch.clamp(original_images, epsilon, 1 - epsilon)


    original_flat = original_images.view(original_images.size(0), -1)
    reconstructed_flat = reconstructed_images.view(reconstructed_images.size(0), -1)

    bce_per_pixel = F.binary_cross_entropy(reconstructed_flat, original_flat, reduction='none')

    reconstruction_error = torch.sum(bce_per_pixel, dim=1)

    return reconstruction_error

def find_optimal_threshold_and_f1(errors, true_labels):
    """
    Finds the optimal threshold for classification based on reconstruction errors
    by maximizing the F1-score.

    Args:
        errors (np.ndarray): Array of reconstruction errors.
        true_labels (np.ndarray): Array of true class labels (0 or 1).

    Returns:
        tuple: (optimal_threshold, best_f1_score)
    """
    best_f1 = 0
    optimal_threshold = -1
    # Check a range of thresholds from min error to max error
    thresholds = np.linspace(np.min(errors), np.max(errors), 100)

    for thresh in thresholds:
        predictions = (errors > thresh).astype(int)
        f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)

        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = thresh

    return optimal_threshold, best_f1


# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- VAE Loss Function ---
def vae_loss_function(reconstructed_x, x, mu, logvar, class_labels, class_weights, beta):
    # Ensure inputs are within [0, 1] range before BCE (as a safeguard)
    epsilon = 1e-6
    reconstructed_x = torch.clamp(reconstructed_x, epsilon, 1 - epsilon)
    x = torch.clamp(x, epsilon, 1 - epsilon)

    # 1. Reconstruction Loss (Binary Cross-Entropy)
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_x.view(-1, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH),
        x.view(-1, IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH),
        reduction='none' # Calculate loss per pixel per image
    )

    # Apply class weights
    weights_for_batch = class_weights[class_labels]
    weighted_reconstruction_loss = reconstruction_loss * weights_for_batch.unsqueeze(1)
    weighted_reconstruction_loss = torch.mean(torch.sum(weighted_reconstruction_loss, dim=1))

    # 2. KL Divergence Loss (Regularization)
    # kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    # Corrected KL divergence calculation based on VAE paper (Kingma & Welling)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) # Average over batch size

    # 3. Total VAE Loss
    total_loss = weighted_reconstruction_loss + beta * kl_divergence

    return total_loss, weighted_reconstruction_loss, kl_divergence

# --- Helper function to save sample images ---
def save_sample_images(model, epoch, device, latent_dim, save_dir='./sample_images'):
    """
    Generates and saves sample images:
    - Reconstructions of a few validation images
    - Images sampled from the latent space prior
    """
    model.eval() # Set model to evaluation mode
    os.makedirs(save_dir, exist_ok=True)

    # Get a fixed batch from validation loader for consistent reconstruction visualization
    # This assumes val_loader is iterable and can be reset or has persistent state if needed
    # For simplicity, let's just take the first batch
    try:
        sample_batch, sample_labels = next(iter(val_loader))
    except StopIteration:
        print("Warning: val_loader is empty, cannot save sample images.")
        return

    sample_batch = sample_batch.to(device)[:min(10, sample_batch.size(0))] # Take up to 10 images
    with torch.no_grad():
        reconstructed_batch, _, _ = model(sample_batch)

        # Generate images from random latent samples
        random_latent_samples = torch.randn(10, latent_dim).to(device)
        decoder_input = model.decoder_input(random_latent_samples)
        decoder_input = decoder_input.view(-1, model.decoder_input.out_features // (4*4), 4, 4) # Reshape dynamically
        generated_samples = model.decoder(decoder_input)


    # Plot and save reconstructions
    fig_rec, axes_rec = plt.subplots(2, sample_batch.size(0), figsize=(sample_batch.size(0) * 2, 4))
    fig_rec.suptitle(f'Epoch {epoch} - Reconstructions')
    for i in range(sample_batch.size(0)):
        # Original Image
        axes_rec[0, i].imshow(sample_batch[i].squeeze().cpu().numpy(), cmap='gray')
        axes_rec[0, i].set_title('Original')
        axes_rec[0, i].axis('off')
        # Reconstructed Image
        axes_rec[1, i].imshow(reconstructed_batch[i].squeeze().cpu().numpy(), cmap='gray')
        axes_rec[1, i].set_title('Recon')
        axes_rec[1, i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_reconstructions.png'))
    plt.close(fig_rec)

    # Plot and save generated images
    fig_gen, axes_gen = plt.subplots(1, generated_samples.size(0), figsize=(generated_samples.size(0) * 2, 2))
    fig_gen.suptitle(f'Epoch {epoch} - Generated Samples')
    for i in range(generated_samples.size(0)):
        axes_gen[i].imshow(generated_samples[i].squeeze().cpu().numpy(), cmap='gray')
        axes_gen[i].set_title(f'Gen {i+1}')
        axes_gen[i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_generated.png'))
    plt.close(fig_gen)

    model.train() # Set model back to training mode


# --- Training Loop (for a single combination) ---
def train_single_combination(model, optimizer, train_loader, val_loader, class_weights, beta, num_epochs, device, latent_dim):
    best_val_f1 = -1 # Track best validation F1-score for this combination
    best_epoch = -1
    best_model_state_dict = None # Store the model state dict for the best epoch

    # Initialize Learning Rate Scheduler
    # Monitor validation F1-score and reduce LR if it doesn't improve for 'patience' epochs
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)


    for epoch in range(1, num_epochs + 1):
        model.train() # Set model to training mode
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            reconstructed_images, mu, logvar = model(images)

            loss, _, _ = vae_loss_function(
                reconstructed_images, images, mu, logvar, labels, class_weights.to(device), beta
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                 print(f'  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item() / images.size(0):.6f}')

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch {epoch} Average Train Loss: {avg_train_loss:.4f}')

        # --- Validation Step (within the training loop) ---
        model.eval() # Set model to evaluation mode
        val_loss = 0
        val_errors = []
        val_true_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                reconstructed_images, mu, logvar = model(images)

                loss, _, _ = vae_loss_function(
                     reconstructed_images, images, mu, logvar, labels, class_weights.to(device), beta
                )
                val_loss += loss.item()

                errors = calculate_reconstruction_error(images, reconstructed_images)
                val_errors.extend(errors.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f'====> Epoch {epoch} Average Validation Loss: {avg_val_loss:.4f}')

        # --- Find Optimal Threshold and F1-score on Validation Set ---
        optimal_threshold, current_val_f1 = find_optimal_threshold_and_f1(np.array(val_errors), np.array(val_true_labels))
        print(f'====> Epoch {epoch} Validation F1-Score: {current_val_f1:.4f} (Threshold: {optimal_threshold:.4f})')

        # Step the learning rate scheduler
        scheduler.step(current_val_f1)


        # Check if this is the best validation F1-score seen so far for this combination
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            best_model_state_dict = model.state_dict().copy()
            print(f"Validation F1-score improved. Best F1 so far: {best_val_f1:.4f} at Epoch {best_epoch}")

        # --- Save sample images periodically ---
        if epoch % SAVE_IMAGE_INTERVAL == 0:
             print(f"Saving sample images for Epoch {epoch}...")
             save_sample_images(model, epoch, device, latent_dim)


    print(f"Finished training for this combination. Best Validation F1: {best_val_f1:.4f} at Epoch {best_epoch}")
    return best_val_f1, best_model_state_dict # Return the best F1 and the corresponding model state

# --- Main Grid Search Execution ---
if __name__ == '__main__':
    best_overall_f1 = -1 # Track the best F1-score across all combinations
    best_hyperparameters = None
    best_model_state = None

    keys = PARAM_GRID.keys()
    combinations = list(itertools.product(*PARAM_GRID.values()))

    print(f"Starting Grid Search with {len(combinations)} combinations.")
    print(f"Hyperparameter Grid: {PARAM_GRID}")

    # Create a directory to save sample images during training
    os.makedirs('./sample_images', exist_ok=True)

    for i, combination in enumerate(combinations):
        hyperparameters = dict(zip(keys, combination))
        lr = hyperparameters['learning_rate']
        latent_dim = hyperparameters['latent_dim']
        beta = hyperparameters['beta']

        print(f"\n--- Training Combination {i+1}/{len(combinations)} ---")
        print(f"Hyperparameters: Learning Rate={lr}, Latent Dim={latent_dim}, Beta={beta}")

        model = VAE(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        current_best_val_f1, current_best_model_state = train_single_combination(
            model, optimizer, train_loader, val_loader, class_weights_tensor,
            beta, NUM_EPOCHS_PER_COMBINATION, device, latent_dim # Pass latent_dim for image saving
        )

        if current_best_val_f1 > best_overall_f1:
            best_overall_f1 = current_best_val_f1
            best_hyperparameters = hyperparameters
            best_model_state = current_best_model_state # Store the state dict of the best model

    print("\n--- Grid Search Finished ---")
    print(f"Best Overall Validation F1-Score: {best_overall_f1:.4f}")
    print(f"Best Hyperparameters: {best_hyperparameters}")

    # Save the state dictionary AND hyperparameters of the overall best model
    if best_model_state is not None:
        final_model_save_path = './best_vae_stroke_model.pth'
        save_dict = {
            'model_state_dict': best_model_state,
            'hyperparameters': best_hyperparameters,
            'best_val_f1': best_overall_f1
        }
        torch.save(save_dict, final_model_save_path)
        print(f"Best model weights and hyperparameters saved to {final_model_save_path}")

    print("\nGrid Search Complete.")
    print("Review the ./sample_images directory for visual progress.")
    print(f"Run evaluate.py using the {final_model_save_path} file for final evaluation and generation.")
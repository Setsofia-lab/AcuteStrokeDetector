import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import os

# Import the VAE model definition
# LATENT_DIM will be loaded from the saved hyperparameters
from model import VAE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH # Removed LATENT_DIM

# Import data loading components (specifically the test loader)
from preprocess import val_loader, test_loader # Assuming class_weights_tensor is not strictly needed here

# --- Configuration ---
# Updated MODEL_PATH to match the new saving format
MODEL_PATH = './best_vae_stroke_model.pth' # Path to the saved model state dict and hyperparameters

# For latent space exploration
NUM_GENERATED_EXAMPLES = 10 # Number of random images to generate
NUM_INTERPOLATION_STEPS = 8 # Number of steps for interpolation visualization
NUM_RECONSTRUCTION_EXAMPLES = 10 # Number of test images to show reconstruction for


# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Instantiate and Load the Trained VAE Model ---
# Load the saved dictionary
if os.path.exists(MODEL_PATH):
    print(f"Loading model weights and hyperparameters from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    loaded_hyperparameters = checkpoint['hyperparameters']
    loaded_model_state_dict = checkpoint['model_state_dict']
    loaded_best_val_f1 = checkpoint['best_val_f1']

    # Get the latent_dim from the loaded hyperparameters
    loaded_latent_dim = loaded_hyperparameters['latent_dim']

    # Instantiate the model with the correct latent dimension
    model = VAE(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, loaded_latent_dim).to(device)

    # Load the state dictionary
    model.load_state_dict(loaded_model_state_dict)
    model.eval() # Set model to evaluation mode

    print(f"Model loaded successfully with latent_dim: {loaded_latent_dim}")
    print(f"Best Validation F1 from training: {loaded_best_val_f1:.4f}")
    print(f"Hyperparameters used: {loaded_hyperparameters}")

else:
    print(f"Error: Model weights and hyperparameters not found at {MODEL_PATH}")
    print("Please run train.py first to train and save the model.")
    exit() # Exit if model not found


# --- Phase 5: Evaluation for Stroke Detection ---

# Function to calculate Reconstruction Error (using BCE per image)
def calculate_reconstruction_error(original_images, reconstructed_images):
    """
    Calculates the Binary Cross-Entropy reconstruction error for each image in a batch.

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

    reconstruction_error = torch.sum(bce_per_pixel, dim=1) # Shape: [batch_size]

    return reconstruction_error

# --- Determine Threshold for Classification ---
print("\nCalculating reconstruction errors on the validation set to determine threshold...")
val_errors = []
val_true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        reconstructed_images, _, _ = model(images) # We only need the reconstruction

        errors = calculate_reconstruction_error(images, reconstructed_images)
        val_errors.extend(errors.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

val_errors = np.array(val_errors)
val_true_labels = np.array(val_true_labels)

print("Finding optimal threshold on validation set using F1-score...")
best_f1 = 0
optimal_threshold = -1
thresholds = np.linspace(np.min(val_errors), np.max(val_errors), 100) # Check 100 thresholds

for thresh in thresholds:
    val_predictions = (val_errors > thresh).astype(int)
    f1 = f1_score(val_true_labels, val_predictions, average='binary', pos_label=1)

    if f1 > best_f1:
        best_f1 = f1
        optimal_threshold = thresh

print(f"Optimal threshold found on validation set: {optimal_threshold:.4f} (F1-score: {best_f1:.4f})")
print("Using this threshold for evaluation on the test set.")


# --- Evaluate on the Test Set ---
print("\nEvaluating model on the test set...")
test_errors = []
test_true_labels = []
test_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        reconstructed_images, _, _ = model(images) # We only need the reconstruction

        errors = calculate_reconstruction_error(images, reconstructed_images)
        test_errors.extend(errors.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

        batch_predictions = (errors.cpu().numpy() > optimal_threshold).astype(int)
        test_predictions.extend(batch_predictions)

test_errors = np.array(test_errors)
test_true_labels = np.array(test_true_labels)
test_predictions = np.array(test_predictions)

# --- Report Evaluation Metrics ---
print("\n--- Evaluation Metrics on Test Set ---")

cm = confusion_matrix(test_true_labels, test_predictions)
print("Confusion Matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

precision = precision_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"Precision (Stroke): {precision:.4f}")

recall = recall_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"Recall (Sensitivity) (Stroke): {recall:.4f}")

f1 = f1_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"F1-Score (Stroke): {f1:.4f}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Overall Accuracy: {accuracy:.4f}")

# Plot ROC Curve
print("\nGenerating ROC curve...")
fpr, tpr, roc_thresholds = roc_curve(test_true_labels, test_errors)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
print("ROC curve displayed.")

# --- Show Original vs. Reconstructed Images ---
print(f"\nDisplaying {NUM_RECONSTRUCTION_EXAMPLES} original vs. reconstructed images from the test set...")

# Get a batch from the test loader
try:
    sample_batch_test, _ = next(iter(test_loader))
except StopIteration:
    print("Warning: test_loader is empty, cannot show reconstructions.")
    sample_batch_test = None

if sample_batch_test is not None:
    sample_batch_test = sample_batch_test.to(device)[:min(NUM_RECONSTRUCTION_EXAMPLES, sample_batch_test.size(0))] # Take up to NUM_RECONSTRUCTION_EXAMPLES

    with torch.no_grad():
        reconstructed_batch_test, _, _ = model(sample_batch_test)

    fig_test_rec, axes_test_rec = plt.subplots(2, sample_batch_test.size(0), figsize=(sample_batch_test.size(0) * 2, 4))
    fig_test_rec.suptitle('Test Set: Original vs. Reconstruction')
    for i in range(sample_batch_test.size(0)):
        # Original Image
        axes_test_rec[0, i].imshow(sample_batch_test[i].squeeze().cpu().numpy(), cmap='gray')
        axes_test_rec[0, i].set_title('Original')
        axes_test_rec[0, i].axis('off')
        # Reconstructed Image
        axes_test_rec[1, i].imshow(reconstructed_batch_test[i].squeeze().cpu().numpy(), cmap='gray')
        axes_test_rec[1, i].set_title('Recon')
        axes_test_rec[1, i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Test set reconstructions displayed.")


# --- Phase 6: Latent Space Exploration ---
print("\n--- Phase 6: Latent Space Exploration ---")

# 1. Generate images by sampling from the latent space prior (Standard Normal)
print(f"\nGenerating {NUM_GENERATED_EXAMPLES} images by sampling from latent space...")
with torch.no_grad():
    random_latent_vectors = torch.randn(NUM_GENERATED_EXAMPLES, loaded_latent_dim).to(device) # Use loaded_latent_dim

    # Pass through the decoder
    decoder_input = model.decoder_input(random_latent_vectors)
    # Reshape - dynamically get the channel size
    decoder_input = decoder_input.view(-1, model.decoder_input.out_features // (4*4), 4, 4)

    generated_images = model.decoder(decoder_input)

print("Displaying generated images...")
fig_gen, axes_gen = plt.subplots(1, NUM_GENERATED_EXAMPLES, figsize=(NUM_GENERATED_EXAMPLES * 2, 2))
fig_gen.suptitle("Images Generated from Latent Space Sampling")
for i in range(NUM_GENERATED_EXAMPLES):
    img = generated_images[i].squeeze().cpu().numpy()
    axes_gen[i].imshow(img, cmap='gray')
    axes_gen[i].axis('off')
    axes_gen[i].set_title(f'Gen {i+1}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("Generated images displayed.")


# 2. Interpolate between two points in the latent space
print(f"\nInterpolating between two test images ({NUM_INTERPOLATION_STEPS} steps)...")
# We need two distinct images from the test set to interpolate between
# Let's try to pick two images (e.g., first non-stroke and first stroke if available)
# This part requires getting specific images, which might be complex depending on your data loading.
# For simplicity, let's just take the first two images from the test loader if available.

interpolation_images = []
interpolation_labels = []
try:
    for images, labels in test_loader:
        interpolation_images.append(images[0]) # Take the first image from each batch
        interpolation_labels.append(labels[0])
        if len(interpolation_images) >= 2:
            break
except StopIteration:
    print("Warning: Could not get at least two images from test_loader for interpolation.")
    interpolation_images = [] # Clear the list if not enough images

if len(interpolation_images) >= 2:
    img1 = interpolation_images[0].unsqueeze(0).to(device) # Add batch dimension
    img2 = interpolation_images[1].unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode the two images to get their latent means
        _, mu1, _ = model(img1)
        _, mu2, _ = model(img2)

        # Perform linear interpolation in the latent space
        interpolated_latent_vectors = []
        for i in range(NUM_INTERPOLATION_STEPS):
            alpha = i / (NUM_INTERPOLATION_STEPS - 1) # Alpha goes from 0 to 1
            interpolated_z = (1 - alpha) * mu1 + alpha * mu2
            interpolated_latent_vectors.append(interpolated_z)

        interpolated_latent_vectors = torch.cat(interpolated_latent_vectors, dim=0) # Combine into a batch

        # Decode the interpolated latent vectors
        decoder_input_interp = model.decoder_input(interpolated_latent_vectors)
        decoder_input_interp = decoder_input_interp.view(-1, model.decoder_input.out_features // (4*4), 4, 4)
        interpolated_images = model.decoder(decoder_input_interp)

    # Display the interpolated images
    print("Displaying interpolated images...")
    fig_interp, axes_interp = plt.subplots(1, NUM_INTERPOLATION_STEPS, figsize=(NUM_INTERPOLATION_STEPS * 2, 2))
    fig_interp.suptitle("Latent Space Interpolation")
    for i in range(NUM_INTERPOLATION_STEPS):
        img = interpolated_images[i].squeeze().cpu().numpy()
        axes_interp[i].imshow(img, cmap='gray')
        axes_interp[i].axis('off')
        axes_interp[i].set_title(f'Step {i+1}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Interpolated images displayed.")
else:
    print("Skipping interpolation as could not get enough test images.")

print("\nEvaluation and Latent Space Exploration Complete.")
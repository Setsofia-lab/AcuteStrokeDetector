import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import os

# Import the VAE model definition
from model import VAE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, LATENT_DIM

# Import data loading components (specifically the test loader)
# We also need the validation loader to help set the threshold
from preprocess import val_loader, test_loader, class_weights_tensor # Assuming class_weights_tensor is needed for threshold analysis

# --- Configuration ---
MODEL_PATH = './vae_stroke_model.pth' # Path to the saved model weights
# For latent space exploration
NUM_GENERATED_EXAMPLES = 10 # Number of random images to generate
NUM_INTERPOLATION_STEPS = 8 # Number of steps for interpolation visualization

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Instantiate and Load the Trained VAE Model ---
model = VAE(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, LATENT_DIM).to(device)

# Check if the model file exists
if os.path.exists(MODEL_PATH):
    print(f"Loading model weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
else:
    print(f"Error: Model weights not found at {MODEL_PATH}")
    print("Please run train.py first to train and save the model.")
    exit() # Exit if model not found

print("Model loaded successfully.")

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
    # Flatten images for BCE calculation
    original_flat = original_images.view(original_images.size(0), -1)
    reconstructed_flat = reconstructed_images.view(reconstructed_images.size(0), -1)

    # Calculate BCE per pixel per image
    # F.binary_cross_entropy returns mean by default, use reduction='none' to get per-element loss
    bce_per_pixel = F.binary_cross_entropy(reconstructed_flat, original_flat, reduction='none')

    # Sum BCE over all pixels for each image to get the total reconstruction error per image
    reconstruction_error = torch.sum(bce_per_pixel, dim=1) # Shape: [batch_size]

    return reconstruction_error

# --- Determine Threshold for Classification ---
# A common approach is to use the validation set to find a good threshold.
# We'll calculate reconstruction errors for all validation images.
print("\nCalculating reconstruction errors on the validation set to determine threshold...")
val_errors = []
val_true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        reconstructed_images, _, _ = model(images) # We only need the reconstruction

        errors = calculate_reconstruction_error(images, reconstructed_images)
        val_errors.extend(errors.cpu().numpy()) # Move to CPU and convert to numpy
        val_true_labels.extend(labels.cpu().numpy())

val_errors = np.array(val_errors)
val_true_labels = np.array(val_true_labels)

# Simple threshold determination: Find the threshold that maximizes the F1-score on the validation set
# This is one way to balance precision and recall, which is important for imbalanced data.
# You could also choose a threshold based on desired recall (e.g., achieve 90% recall).
print("Finding optimal threshold on validation set using F1-score...")
best_f1 = 0
optimal_threshold = -1
thresholds = np.linspace(np.min(val_errors), np.max(val_errors), 100) # Check 100 thresholds

for thresh in thresholds:
    # Classify validation data based on the current threshold
    # Prediction is 1 (stroke) if error > threshold, 0 (non-stroke) otherwise
    val_predictions = (val_errors > thresh).astype(int)

    # Calculate F1-score for the current threshold
    # Use 'binary' for binary classification, 'pos_label=1' specifies stroke as the positive class
    f1 = f1_score(val_true_labels, val_predictions, average='binary', pos_label=1)

    # Check if this threshold gives a better F1-score
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

        # Classify based on the optimal threshold
        batch_predictions = (errors.cpu().numpy() > optimal_threshold).astype(int)
        test_predictions.extend(batch_predictions)

test_errors = np.array(test_errors)
test_true_labels = np.array(test_true_labels)
test_predictions = np.array(test_predictions)

# --- Report Evaluation Metrics ---
print("\n--- Evaluation Metrics on Test Set ---")

# Confusion Matrix
# Rows are true labels, columns are predicted labels
# [[TN, FP], [FN, TP]]
cm = confusion_matrix(test_true_labels, test_predictions)
print("Confusion Matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel() # Unpack the confusion matrix values
print(f"True Negatives (TN): {tn}") # Correctly identified non-stroke
print(f"False Positives (FP): {fp}") # Incorrectly identified non-stroke as stroke (Type I error)
print(f"False Negatives (FN): {fn}") # Incorrectly identified stroke as non-stroke (Type II error - critical!)
print(f"True Positives (TP): {tp}") # Correctly identified stroke

# Precision: Of all predicted stroke cases, how many were actually stroke?
# Precision = TP / (TP + FP)
precision = precision_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"Precision (Stroke): {precision:.4f}")

# Recall (Sensitivity): Of all actual stroke cases, how many were correctly identified?
# Recall = TP / (TP + FN)
# This is a very important metric for medical diagnosis - we want high recall to minimize missed stroke cases.
recall = recall_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"Recall (Sensitivity) (Stroke): {recall:.4f}")

# F1-Score: Harmonic mean of Precision and Recall. Balances both.
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
f1 = f1_score(test_true_labels, test_predictions, average='binary', pos_label=1)
print(f"F1-Score (Stroke): {f1:.4f}")

# Accuracy (less informative for imbalanced data, but good to see)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Overall Accuracy: {accuracy:.4f}")

# Optional: Plot ROC Curve (Receiver Operating Characteristic)
# Shows the trade-off between True Positive Rate (Recall) and False Positive Rate at various thresholds.
# fpr = FP / (FP + TN)
# tpr = TP / (TP + FN) = Recall
print("\nGenerating ROC curve...")
# We need the reconstruction errors and true labels to calculate ROC curve points
fpr, tpr, roc_thresholds = roc_curve(test_true_labels, test_errors) # Note: roc_curve expects higher scores for positive class, so using errors directly works
roc_auc = auc(fpr, tpr) # Area Under the ROC Curve - higher is better

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
print("ROC curve displayed.")


# --- Phase 6: Latent Space Exploration ---
print("\n--- Phase 6: Latent Space Exploration ---")

# 1. Generate images by sampling from the latent space prior (Standard Normal)
print(f"\nGenerating {NUM_GENERATED_EXAMPLES} images by sampling from latent space...")
with torch.no_grad():
    # Sample random latent vectors from the standard normal distribution
    # Shape: [NUM_GENERATED_EXAMPLES, LATENT_DIM]
    random_latent_vectors = torch.randn(NUM_GENERATED_EXAMPLES, LATENT_DIM).to(device)

    # Pass the latent vectors through the decoder to generate images
    # The decoder expects input shaped like [batch_size, flattened_size] after the first linear layer
    # and then reshaped. Let's call the decoder directly after the first linear layer.
    # Need to replicate the first part of the decoder's forward pass
    decoder_input = model.decoder_input(random_latent_vectors)
    # Reshape to the spatial dimensions expected by the ConvTranspose layers
    # This size matches the output of the last conv layer before flatten in the encoder
    # For 64x64 example, this is [batch_size, 256, 4, 4]
    decoder_input = decoder_input.view(-1, 256, 4, 4) # Use hardcoded 256 and 4x4 based on model architecture

    generated_images = model.decoder(decoder_input) # Pass through the ConvTranspose layers

# Visualize generated images
print("Displaying generated images...")
fig, axes = plt.subplots(1, NUM_GENERATED_EXAMPLES, figsize=(NUM_GENERATED_EXAMPLES * 2, 2))
for i in range(NUM_GENERATED_EXAMPLES):
    # Move image to CPU and convert to numpy, remove channel dimension
    img = generated_images[i].squeeze().cpu().numpy()
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off') # Hide axes
    axes[i].set_title(f'Gen {i+1}')
plt.suptitle("Images Generated from Latent Space Sampling")
plt.show()
print("Generated images displayed.")


# 2. Interpolate between two points in the latent space
print(f"\nInterpolating between two test images ({NUM_INTERPOLATION_STEPS} steps)...")

# Get two images from the test set (e.g., the first two)
# Ensure test_loader has at least 2 images and batch size >= 2
try:
    # Get the first batch from the test loader
    test_images_batch, _ = next(iter(test_loader))
    if test_images_batch.size(0) < 2:
         print("Test loader batch size is less than 2. Cannot perform interpolation.")
    else:
        # Take the first two images from the batch
        img1 = test_images_batch[0].unsqueeze(0).to(device) # Add batch dimension
        img2 = test_images_batch[1].unsqueeze(0).to(device) # Add batch dimension

        # Get the latent representations (mean vectors) for these two images
        with torch.no_grad():
            _, mu1, _ = model(img1)
            _, mu2, _ = model(img2)

        # Perform linear interpolation in the latent space
        # Interpolated latent vector = (1 - alpha) * mu1 + alpha * mu2
        # alpha goes from 0 to 1
        interpolated_latent_vectors = []
        for i in range(NUM_INTERPOLATION_STEPS):
            alpha = i / (NUM_INTERPOLATION_STEPS - 1)
            interp_z = (1 - alpha) * mu1 + alpha * mu2
            interpolated_latent_vectors.append(interp_z)

        # Concatenate the interpolated vectors into a single tensor
        interpolated_latent_vectors = torch.cat(interpolated_latent_vectors, dim=0).to(device)

        # Pass the interpolated latent vectors through the decoder
        with torch.no_grad():
            decoder_input_interp = model.decoder_input(interpolated_latent_vectors)
            decoder_input_interp = decoder_input_interp.view(-1, 256, 4, 4) # Reshape
            interpolated_images = model.decoder(decoder_input_interp)

        # Visualize interpolated images
        print("Displaying interpolated images...")
        fig, axes = plt.subplots(1, NUM_INTERPOLATION_STEPS, figsize=(NUM_INTERPOLATION_STEPS * 2, 2))
        for i in range(NUM_INTERPOLATION_STEPS):
            img = interpolated_images[i].squeeze().cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Step {i+1}')
        plt.suptitle("Latent Space Interpolation")
        plt.show()
        print("Interpolated images displayed.")

except StopIteration:
    print("Test loader is empty. Cannot perform interpolation.")
except Exception as e:
    print(f"Error during interpolation: {e}")


print("\nPhase 5 & 6: Evaluation and Latent Space Exploration Complete.")

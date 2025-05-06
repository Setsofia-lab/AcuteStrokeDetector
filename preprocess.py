import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image # Using Pillow for image processing
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms # Using torchvision transforms
import torch.nn.functional as F # Import F for clamp

# Define directories for data
DATA_DIR = './data/input'
STROKE_DIR = os.path.join(DATA_DIR, 'stroke_data')
NON_STROKE_DIR = os.path.join(DATA_DIR, 'noStroke_data')

# Image dimensions for resizing - Make sure these match model.py
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# DataLoader parameters
BATCH_SIZE = 32
NUM_WORKERS = 3 # Adjust based on your system's CPU cores

# --- 1. Load Data and Labels ---
print("Loading data and labels...")
all_image_paths = []
all_labels = [] # 1 for stroke, 0 for non-stroke

# Load stroke data with label 1
if os.path.exists(STROKE_DIR):
    # Filter for common image file extensions
    stroke_image_files = [os.path.join(STROKE_DIR, f) for f in os.listdir(STROKE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    all_image_paths.extend(stroke_image_files)
    all_labels.extend([1] * len(stroke_image_files))
    print(f"Found {len(stroke_image_files)} stroke images.")
else:
    print(f"Stroke directory not found: {STROKE_DIR}. Please ensure your data is in this directory structure.")

# Load non-stroke data with label 0
if os.path.exists(NON_STROKE_DIR):
    # Filter for common image file extensions
    non_stroke_image_files = [os.path.join(NON_STROKE_DIR, f) for f in os.listdir(NON_STROKE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    all_image_paths.extend(non_stroke_image_files)
    all_labels.extend([0] * len(non_stroke_image_files))
    print(f"Found {len(non_stroke_image_files)} non-stroke images.")
else:
    print(f"Non-stroke directory not found: {NON_STROKE_DIR}. Please ensure your data is in this directory structure.")


if len(all_image_paths) == 0:
    print("Error: No images found in the specified data directories.")
    exit()

# Convert to numpy arrays for easier handling
all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

print(f"Total images found: {len(all_image_paths)}")
print(f"Total labels found: {len(all_labels)}")


# --- 2. Data Imbalance using Weighted Loss Calculation ---
class_counts = np.bincount(all_labels)
num_classes = len(class_counts)
total_samples = len(all_labels)

# Calculate weights: Total Samples / (Number of Classes * Samples in Class)
# This gives less weight to the majority class and more to the minority class
class_weights = total_samples / (num_classes * class_counts)

# Convert to torch tensor for use in loss function later
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"\nClass counts (0: Non-Stroke, 1: Stroke): {class_counts}")
print(f"Calculated class weights (0: Non-Stroke, 1: Stroke): {class_weights}")
print("These weights will be used in the VAE loss calculation in train.py to penalize errors on the stroke class more.")


# --- 3. Resampling Discussion (as per your note) ---
# print("\nResampling Discussion:")
# print("Resampling (like oversampling the minority class or undersampling the majority class)")
# print("is another technique to address data imbalance. For this project's timeframe,")
# print("we are primarily relying on the weighted loss approach, which is often simpler")
# print("to implement within the loss function itself.")
# print("If more time were available, exploring techniques like RandomOverSampler or")
# print("SMOTE from the 'imblearn' library could be beneficial.")


# --- 4. Split Data into train, validation, test set ---
print("\nSplitting data into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(
    all_image_paths, all_labels,
    train_size=TRAIN_RATIO,
    random_state=42,
    stratify=all_labels # ensures class distribution is similar in splits
)

# Then, split temp into validation and test.
# Calculate the ratio for the second split relative to the temp set size
val_ratio_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=1 - val_ratio_temp, # Use the calculated ratio for test size
    random_state=42,
    stratify=y_temp # stratify the temp set to maintain distribution
)

print(f"Train set size: {len(X_train)} images, {np.bincount(y_train)} class distribution")
print(f"Validation set size: {len(X_val)} images, {np.bincount(y_val)} class distribution")
print(f"Test set size: {len(X_test)} images, {np.bincount(y_test)} class distribution")


# --- 5. Image Preprocessing Function ---
def preprocess_image(image_path):
    """
    Loads, preprocesses, and converts an image to a tensor.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        torch.Tensor: The preprocessed image tensor (shape [1, H, W]), or None if processing fails.
    """
    try:
        # Load image using Pillow and convert to grayscale ('L')
        img = Image.open(image_path).convert('L')

        # Resize image to consistent dimensions
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to tensor and normalize (0-255 to 0-1)
        # torchvision.transforms.ToTensor() handles this conversion and normalization
        # It also changes the shape from (H, W) to (C, H, W), where C=1 for grayscale
        img_tensor = transforms.ToTensor()(img)

        # --- Potential point for Facial Alignment ---
        # If you implement facial alignment, this is where you would add the code
        # after loading and before converting to tensor. Example (conceptual):
        # aligned_img = align_face(img) # Your alignment function
        # img_tensor = transforms.ToTensor()(aligned_img) # Convert aligned image

        # Clamp values to a small epsilon range to avoid log(0) or log(1) issues in BCE
        # Consistent with the clamp added in the loss function
        epsilon = 1e-6
        img_tensor = torch.clamp(img_tensor, epsilon, 1 - epsilon)


        return img_tensor

    except Exception as e:
        # Catch and print errors during image processing for debugging
        print(f"Error processing image {image_path}: {e}")
        return None # Return None if processing fails


# --- Define Custom Dataset Class ---
class StrokeDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Return one sample (image tensor and label) at the given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Preprocess the image using the function defined above
        image_tensor = preprocess_image(img_path)

        # Handle cases where image processing might fail (e.g., corrupted file)
        # This is important when using num_workers, as errors can be cryptic
        if image_tensor is None:
             # You might want to skip this index or handle it differently
             # For simplicity here, we'll raise an error, but a more robust
             # approach might involve filtering out bad paths beforehand
             raise RuntimeError(f"Failed to load and preprocess image at index {idx}: {img_path}")


        return image_tensor, label


# --- 6. Create Data Loaders with Parallelism ---
print("\nCreating DataLoaders...")

# Create dataset instances for each split
train_dataset = StrokeDataset(X_train, y_train)
val_dataset = StrokeDataset(X_val, y_val)
test_dataset = StrokeDataset(X_test, y_test)

# Create DataLoader instances
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle training data
    num_workers=NUM_WORKERS,
    pin_memory=True, # Speeds up data transfer to GPU if using CUDA
    # Add a collate_fn if you were handling None results from __getitem__ by filtering
    # collate_fn=lambda x: tuple(zip(*x)) # Example collate_fn if filtering Nones
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle test data
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Train loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print(f"Validation loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print(f"Test loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

print("\nPreprocessing setup complete.")
print(f"The class weights tensor for the loss function is: {class_weights_tensor.tolist()}")
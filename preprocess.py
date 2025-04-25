# 1-  Firstly load the data using os and adding each label to it, since the data has stroke and non_stroke images, need to use python os module make a list of file poaths and their corresponing labels
# 2 - For the imbalanced datset issue, I want to consider Weighted loss to penealise the majority class more - so this would account for the imbalanced dataset
# 3 - Also explore resampling to chose more of the stroke dataset since its underpresented and less of the non_stroke data during model traing.
# 4 - Splitng into train, validation, test sets

# Image preprocessing steps
# 1.convert to grayscale to reduce the size of the image
# 2. resize all image dimesions to a static 64x64 for consistent image dimesions accross
# 3.normalise the images from 0-255 pixels to 0-1 , for faster preprocessing
# 4. convert the images to tensor with 1 channel dimesion

# Create data loader to load and feed batches of the data to the model during training, include parallelism for faster processing and use of more CPUs

import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image # Using Pillow for image processing
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms # Using torchvision transforms


DATA_DIR = './data'
STROKE_DIR = os.path.join(DATA_DIR, 'stroke_data')
NON_STROKE_DIR = os.path.join(DATA_DIR, 'noStroke_data')

# Image dimensions for resizing
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# DataLoader parameters
BATCH_SIZE = 32
NUM_WORKERS = 3

#  Load Data and Labels 
print("Loading data and labels")
all_image_paths = []
all_labels = [] # 1 for stroke, 0 for non-stroke

# Load stroke data with label 1
if os.path.exists(STROKE_DIR):
    stroke_image_files = [os.path.join(STROKE_DIR, f) for f in os.listdir(STROKE_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    all_image_paths.extend(stroke_image_files)
    all_labels.extend([1] * len(stroke_image_files))
    print(f"Found {len(stroke_image_files)} stroke images.")
else:
    print(f"Stroke directory not found: {STROKE_DIR}")

# Load non-stroke data with label 0
if os.path.exists(NON_STROKE_DIR):
    non_stroke_image_files = [os.path.join(NON_STROKE_DIR, f) for f in os.listdir(NON_STROKE_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    all_image_paths.extend(non_stroke_image_files)
    all_labels.extend([0] * len(non_stroke_image_files))
    print(f"Found {len(non_stroke_image_files)} non-stroke images.")
else:
    print(f"Non-stroke directory not found: {NON_STROKE_DIR}")

print(f"Total images found: {len(all_image_paths)}")
print(f"Total labels found: {len(all_labels)}")

if len(all_image_paths) == 0:
    print("No images found.")
    exit() 

# Convert to numpy arrays for easier handling
all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

#  Data Imbalance usinf weighted loss calculation. Calculate class counts
class_counts = np.bincount(all_labels)
num_classes = len(class_counts)
total_samples = len(all_labels)

# Calculate weights for the loss function Weight for a class = Total Samples / (Number of Classes * Samples in Class) .This gives less weight to the majority class and more to the minority class
class_weights = total_samples / (num_classes * class_counts)

# Convert to torch tensor for use in loss function later
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"\nClass counts: {class_counts}")
print(f"Calculated class weights for loss function 0 for Non-Stroke, 1 for Stroke: {class_weights}")
print("These weights will be used in the VAE loss calculation to penalize errors on the stroke class more.")

#  Resampling Discussion (Conceptual for now)
# print("\nResampling Discussion:")
# print("Resampling (like oversampling the minority class or undersampling the majority class)")
# print("is another technique to address data imbalance. For this project's timeframe,")
# print("we are primarily relying on the weighted loss approach, which is often simpler")
# print("to implement within the loss function itself.")
# print("If more time were available, exploring techniques like RandomOverSampler or")
# print("SMOTE from the 'imblearn' library could be beneficial.")

#  Split Data into train, validation, test set
print("\nSplitting data into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(
    all_image_paths, all_labels,
    train_size=TRAIN_RATIO,
    random_state=42,
    stratify=all_labels # ensures class distribution is similar in splits
)

# Then, split temp into validation and test. Calculate the ratio for the second split relative to the temp set size . val_ratio_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), # Calculate test size relative to temp
    random_state=42,
    stratify=y_temp 
)

print(f"Train set size: {len(X_train)} images, {np.bincount(y_train)} class distribution")
print(f"Validation set size: {len(X_val)} images, {np.bincount(y_val)} class distribution")
print(f"Test set size: {len(X_test)} images, {np.bincount(y_test)} class distribution")

# Image Preprocessing Function 
# This function will be called by the custom Dataset class
def preprocess_image(image_path):
    """
    Loads, preprocesses, and converts an image to a tensor.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    try:
        # load image using Pillow
        img = Image.open(image_path).convert('L') 

        # resize image
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))

        # convert to tensor and normalize (0-255 to 0-1) torchvision.transforms.ToTensor() does this automatically. It also changes the shape from (H, W) to (C, H, W), where C=1 for grayscale
        img_tensor = transforms.ToTensor()(img)

        #  check that the tensor has a channel dimension Totensor will add it. 
        # img_tensor = img_tensor.unsqueeze(0) 
        return img_tensor

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None # Return None if processing fails

# This StrokedatSet class will decide how to load individual samples
class StrokeDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        # return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # return one sample (image and label) at the given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # preprocess the image using the function defined above
        image_tensor = preprocess_image(img_path)
        return image_tensor, label

#  data loaders with paralellism 
print("\nCreating DataLoaders")

# dataset instances for each split
train_dataset = StrokeDataset(X_train, y_train)
val_dataset = StrokeDataset(X_val, y_val)
test_dataset = StrokeDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # Speeds up data transfer to GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Train loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print(f"Validation loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print(f"Test loader created with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
print("\nPreprocessing complete.")
print(f"The class weights tensor for the loss function is': {class_weights_tensor.tolist()}")


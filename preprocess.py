import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for containerized environment
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import albumentations as A
import multiprocessing as mp
from pathlib import Path
import random
import argparse
import json
from functools import partial
import time

def load_dataset(data_dir):
    """
    Load images and labels from the dataset directory structure.
    
    Args:
        data_dir: Path to the dataset directory with subdirectories for each class
        
    Returns:
        images: List of images as numpy arrays
        labels: List of corresponding labels
    """
    print("Loading dataset...")
    images = []
    labels = []
    
    # Define class directories
    stroke_dir = os.path.join(data_dir, '/Users/samuelsetsofia/dev/projects/AcuteStrokeDetector/data/input/stroke_data')
    non_stroke_dir = os.path.join(data_dir, '/Users/samuelsetsofia/dev/projects/AcuteStrokeDetector/data/input/noStroke_data')
    
    # Function to load a single image
    def load_image(img_path):
        img = cv2.imread(img_path)
        return img if img is not None else None
    
    # Function to load images from a directory in parallel
    def load_directory_images(directory, label, num_processes):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist!")
            return [], []
            
        img_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_files:
            print(f"Warning: No images found in {directory}")
            return [], []
        
        with mp.Pool(processes=num_processes) as pool:
            loaded_images = list(tqdm(
                pool.imap(load_image, img_files),
                total=len(img_files),
                desc=f"Loading {'stroke' if label == 1 else 'non-stroke'} images"
            ))
        
        # Filter out None values (failed loads)
        valid_images = [img for img in loaded_images if img is not None]
        valid_labels = [label] * len(valid_images)
        
        return valid_images, valid_labels
    
    # Determine number of processes to use (leave one core free)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores for loading images")
    
    # Load stroke images in parallel
    stroke_images, stroke_labels = load_directory_images(stroke_dir, 1, num_processes)
    images.extend(stroke_images)
    labels.extend(stroke_labels)
    
    # Load non-stroke images in parallel
    non_stroke_images, non_stroke_labels = load_directory_images(non_stroke_dir, 0, num_processes)
    images.extend(non_stroke_images)
    labels.extend(non_stroke_labels)
    
    return images, labels

def process_image_batch(batch, process_func):
    """Process a batch of images with the given function"""
    return [process_func(img) for img in batch]

def parallel_process_images(images, process_func, description="Processing"):
    """
    Process images in parallel across CPU cores.
    
    Args:
        images: List of images to process
        process_func: Function to apply to each image
        description: Description for the progress bar
        
    Returns:
        processed_images: List of processed images
    """
    # Determine number of processes (leave one core free)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores for {description}")
    
    # Split images into batches for each process
    batches = np.array_split(images, num_processes)
    
    # Create partial function with the process_func
    batch_func = partial(process_image_batch, process_func=process_func)
    
    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(batch_func, batches),
            total=len(batches),
            desc=description
        ))
    
    # Flatten results
    processed_images = [img for batch in results for img in batch]
    
    return processed_images

def resize_normalize_images(images, target_size=(224, 224)):
    """
    Resize images to target size and normalize pixel values.
    
    Args:
        images: List of images as numpy arrays
        target_size: Tuple of (height, width) to resize images to
        
    Returns:
        processed_images: Numpy array of processed images
    """
    print("Resizing and normalizing images...")
    
    def process_single_image(img):
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img_resized = cv2.resize(img_rgb, target_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized / 255.0
        
        return img_normalized
    
    processed_images = parallel_process_images(
        images, 
        process_single_image, 
        "Resizing and normalizing"
    )
    
    return np.array(processed_images)

def analyze_dataset(images, labels):
    """
    Analyze and print dataset statistics.
    
    Args:
        images: List of images as numpy arrays
        labels: List of corresponding labels
    """
    labels_array = np.array(labels)
    stroke_count = np.sum(labels_array == 1)
    non_stroke_count = np.sum(labels_array == 0)
    
    print(f"Total images: {len(images)}")
    print(f"Stroke images: {stroke_count} ({stroke_count/len(images)*100:.2f}%)")
    print(f"Non-stroke images: {non_stroke_count} ({non_stroke_count/len(images)*100:.2f}%)")
    
    # Calculate mean and standard deviation
    all_images = np.array(images)
    mean_per_channel = np.mean(all_images, axis=(0, 1, 2))
    std_per_channel = np.std(all_images, axis=(0, 1, 2))
    
    print(f"Mean per channel: {mean_per_channel}")
    print(f"Std per channel: {std_per_channel}")
    
    # Check image dimensions
    dimensions = [img.shape for img in images[:100]]  # Check first 100 images
    unique_dims = set(dimensions)
    print(f"Image dimensions found: {unique_dims}")
    
    # Return analysis data for saving
    return {
        "total_images": len(images),
        "stroke_images": int(stroke_count),
        "non_stroke_images": int(non_stroke_count),
        "stroke_percentage": float(stroke_count/len(images)*100),
        "non_stroke_percentage": float(non_stroke_count/len(images)*100),
        "mean_per_channel": mean_per_channel.tolist(),
        "std_per_channel": std_per_channel.tolist(),
        "unique_dimensions": [list(dim) for dim in unique_dims]
    }

def create_augmentation_pipeline():
    """
    Create an augmentation pipeline for data augmentation.
    
    Returns:
        A.Compose object with augmentation transforms
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    ])

def augment_dataset(images, labels, augmentation_pipeline, multiplier=2):
    """
    Augment the dataset to increase its size in parallel.
    
    Args:
        images: List of original images
        labels: List of original labels
        augmentation_pipeline: Albumentation pipeline for augmentation
        multiplier: Number of augmented samples to generate per original image
        
    Returns:
        augmented_images: List containing original and augmented images
        augmented_labels: List containing original and augmented labels
    """
    print("Augmenting dataset...")
    augmented_images = images.copy()
    augmented_labels = labels.copy()
    
    # Prepare augmentation task that will be executed in parallel
    def augment_image_set(image_label_pairs):
        aug_images = []
        aug_labels = []
        
        for img, label in image_label_pairs:
            for _ in range(multiplier):
                augmented = augmentation_pipeline(image=img)
                augmented_img = augmented['image']
                aug_images.append(augmented_img)
                aug_labels.append(label)
        
        return aug_images, aug_labels
    
    # Determine number of processes (leave one core free)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} CPU cores for augmentation")
    
    # Create image-label pairs
    image_label_pairs = list(zip(images, labels))
    
    # Split into batches for each process
    batch_size = len(image_label_pairs) // num_processes
    if batch_size == 0:
        batch_size = 1
    
    batches = [image_label_pairs[i:i+batch_size] for i in range(0, len(image_label_pairs), batch_size)]
    
    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(augment_image_set, batches),
            total=len(batches),
            desc="Augmenting images"
        ))
    
    # Combine results
    for aug_images, aug_labels in results:
        augmented_images.extend(aug_images)
        augmented_labels.extend(aug_labels)
    
    return augmented_images, augmented_labels

def standardize_images(images, mean, std):
    """
    Standardize images using mean and standard deviation.
    
    Args:
        images: Numpy array of images
        mean: Mean value per channel
        std: Standard deviation per channel
        
    Returns:
        standardized_images: Numpy array of standardized images
    """
    print("Standardizing images...")
    
    def standardize_single_image(img):
        # Z-score normalization: (x - μ) / σ
        return (img - mean) / std
    
    standardized_images = parallel_process_images(
        images, 
        standardize_single_image, 
        "Standardizing"
    )
    
    return np.array(standardized_images)

def create_train_val_test_split(images, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        images: Numpy array of images
        labels: Numpy array of labels
        test_size: Proportion of the dataset to include in the test split
        val_size: Proportion of the training dataset to include in the validation split
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_val, X_test: Training, validation, and test images
        y_train, y_val, y_test: Training, validation, and test labels
    """
    print("Splitting dataset into train, validation, and test sets...")
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation set from remaining data
    # Adjusted validation size to account for the test split
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def balance_classes(images, labels):
    """
    Balance classes by undersampling the majority class.
    
    Args:
        images: List of images
        labels: List of labels
        
    Returns:
        balanced_images: List of images with balanced classes
        balanced_labels: List of labels with balanced classes
    """
    print("Balancing classes...")
    labels_array = np.array(labels)
    stroke_indices = np.where(labels_array == 1)[0]
    non_stroke_indices = np.where(labels_array == 0)[0]
    
    # Get counts
    stroke_count = len(stroke_indices)
    non_stroke_count = len(non_stroke_indices)
    
    if stroke_count == non_stroke_count:
        print("Classes are already balanced")
        return images, labels
    
    # Determine which class is the majority
    if stroke_count > non_stroke_count:
        majority_indices = stroke_indices
        minority_indices = non_stroke_indices
        majority_count = stroke_count
        minority_count = non_stroke_count
    else:
        majority_indices = non_stroke_indices
        minority_indices = stroke_indices
        majority_count = non_stroke_count
        minority_count = stroke_count
    
    # Randomly select from the majority class to match minority class size
    selected_majority_indices = np.random.choice(
        majority_indices, size=minority_count, replace=False
    )
    
    # Combine minority and selected majority indices
    balanced_indices = np.concatenate([minority_indices, selected_majority_indices])
    
    # Shuffle to mix the classes
    np.random.shuffle(balanced_indices)
    
    # Create balanced dataset
    balanced_images = [images[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    
    print(f"Balanced dataset: {len(balanced_images)} images total")
    
    return balanced_images, balanced_labels

def apply_histogram_equalization(images):
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        images: List of RGB images
        
    Returns:
        equalized_images: List of images with histogram equalization applied
    """
    print("Applying histogram equalization...")
    
    def equalize_single_image(img):
        # Convert to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Apply histogram equalization to the V channel
        img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
        
        # Convert back to RGB
        img_equalized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        return img_equalized
    
    equalized_images = parallel_process_images(
        images, 
        equalize_single_image, 
        "Equalizing histograms"
    )
    
    return equalized_images

def save_sample_images(images, labels, output_dir, class_names=None, num_samples=5):
    """
    Save sample images from the dataset.
    
    Args:
        images: List of images
        labels: List of corresponding labels
        output_dir: Directory to save images
        class_names: List of class names
        num_samples: Number of samples to save
    """
    if class_names is None:
        class_names = ['Non-Stroke', 'Acute Stroke']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Randomly select samples
    indices = np.random.choice(range(len(images)), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[idx])
        plt.title(f"Class: {class_names[labels[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"), dpi=300)
    plt.close()

def extract_symmetric_features(images):
    """
    Extract and enhance features related to facial symmetry.
    
    Args:
        images: List of face images
        
    Returns:
        symmetry_features: List of enhanced symmetry features
    """
    print("Extracting symmetry features...")
    
    # Setup face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_symmetry_from_single_image(img):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Take the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Extract face region
            face_img = img[y:y+h, x:x+w]
            
            # Find the midline
            midline = w // 2
            
            # Split face into left and right halves
            left_half = face_img[:, :midline]
            right_half = face_img[:, midline:]
            
            # Flip the right half to match orientation of left half
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize if needed to ensure both halves are same size
            if left_half.shape[1] != right_half_flipped.shape[1]:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            # Compute absolute difference between left and right halves
            diff_map = cv2.absdiff(left_half, right_half_flipped)
            
            # Enhance the difference map
            diff_map_enhanced = cv2.convertScaleAbs(diff_map, alpha=2.0, beta=0)
            
            # Create a symmetric feature map
            sym_feature = np.zeros_like(img)
            sym_feature[:h, x:x+w] = cv2.cvtColor(diff_map_enhanced, cv2.COLOR_RGB2BGR)
            
            return sym_feature
        else:
            # If no face detected, use the original image
            return img
    
    symmetry_features = parallel_process_images(
        images, 
        extract_symmetry_from_single_image, 
        "Processing symmetry"
    )
    
    return symmetry_features

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, mean, std):
    """
    Save preprocessed data to output directory.
    
    Args:
        X_train, X_val, X_test: Training, validation, and test images
        y_train, y_val, y_test: Training, validation, and test labels
        output_dir: Directory to save data
        mean, std: Mean and standard deviation used for standardization
    """
    print("Saving preprocessed data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Save mean and std
    np.save(os.path.join(output_dir, "mean.npy"), mean)
    np.save(os.path.join(output_dir, "std.npy"), std)
    
    # Save metadata
    metadata = {
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_val_shape": y_val.shape,
        "y_test_shape": y_test.shape,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "preprocessing_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Data saved to {output_dir}")

def preprocess_acute_stroke_dataset(data_dir, output_dir, target_size=(224, 224), augment=True, balance=True):
    """
    Complete preprocessing pipeline for acute stroke dataset.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to save output
        target_size: Target size for resizing images
        augment: Whether to apply data augmentation
        balance: Whether to balance classes
        
    Returns:
        X_train, X_val, X_test: Preprocessed image data
        y_train, y_val, y_test: Labels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dataset in parallel
    images, labels = load_dataset(data_dir)
    
    if not images:
        print("No images were loaded. Check your data directory structure.")
        return None, None, None, None, None, None, None, None
    
    # 2. Analyze dataset
    analysis_data = analyze_dataset(images, labels)
    
    # Save analysis data
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        json.dump(analysis_data, f, indent=4)
    
    # 3. Save sample images before preprocessing
    save_sample_images(images, labels, os.path.join(output_dir, "original_samples"), num_samples=5)
    
    # 4. Resize images to uniform size and convert to RGB
    def initial_process(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        return img_resized
    
    processed_images = parallel_process_images(
        images, 
        initial_process, 
        "Initial processing"
    )
    
    # 5. Apply histogram equalization to enhance contrast
    equalized_images = apply_histogram_equalization(processed_images)
    
    # 6. Balance classes if requested
    if balance:
        balanced_images, balanced_labels = balance_classes(equalized_images, labels)
    else:
        balanced_images, balanced_labels = equalized_images, labels
    
    # 7. Apply data augmentation if requested
    if augment:
        augmentation_pipeline = create_augmentation_pipeline()
        augmented_images, augmented_labels = augment_dataset(
            balanced_images, balanced_labels, augmentation_pipeline
        )
    else:
        augmented_images, augmented_labels = balanced_images, balanced_labels
    
    # 8. Extract symmetry features (optional for stroke detection)
    symmetry_enhanced_images = extract_symmetric_features(augmented_images)
    
    # 9. Normalize pixel values to [0, 1]
    def normalize_image(img):
        return img / 255.0
    
    normalized_images = np.array(parallel_process_images(
        symmetry_enhanced_images,
        normalize_image,
        "Normalizing"
    ))
    
    # 10. Calculate dataset statistics for standardization
    mean = np.mean(normalized_images, axis=(0, 1, 2))
    std = np.std(normalized_images, axis=(0, 1, 2))
    
    # 11. Apply standardization
    standardized_images = standardize_images(normalized_images, mean, std)
    
    # 12. Split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        standardized_images, np.array(augmented_labels)
    )
    
    # 13. Convert labels to categorical format
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)
    
    # 14. Save samples after preprocessing
    save_sample_images(X_train, np.argmax(y_train, axis=1), 
                     os.path.join(output_dir, "processed_samples"), num_samples=5)
    
    # 15. Save preprocessed data
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir, mean, std)
    
    print("Preprocessing complete!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mean, std

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess acute stroke dataset')
    parser.add_argument('--input', type=str, default='/data/input',
                        help='Input directory containing the dataset')
    parser.add_argument('--output', type=str, default='/data/output',
                        help='Output directory to save preprocessed data')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                        help='Target size for images (height width)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable class balancing')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=== Preprocessing Configuration ===")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target size: {args.target_size}")
    print(f"Data augmentation: {not args.no_augment}")
    print(f"Class balancing: {not args.no_balance}")
    print("=================================")
    
    # Run preprocessing
    preprocess_acute_stroke_dataset(
        data_dir=args.input,
        output_dir=args.output,
        target_size=tuple(args.target_size),
        augment=not args.no_augment,
        balance=not args.no_balance
    )
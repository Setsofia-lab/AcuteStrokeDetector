# Acute Stroke Detection using Variational Autoencoders (VAE) and Facial Asymmetry Analysis

## Project Goal

The primary goal of this project is to build and train a Variational Autoencoder (VAE) to learn a compressed representation of human faces. Leveraging the VAE's ability to create a smooth latent space, we aim to explore this space to potentially identify dimensions controlling facial symmetry. The ultimate objective is to generate images that mimic facial drooping, a common symptom of acute stroke, which could aid in early diagnosis and understanding.

## Dataset

This project utilizes the **Kaggle Acute Stroke dataset**: [https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke](https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke)

* **Content:** Contains images of faces labeled as either having acute stroke or not having acute stroke.
* **Key Challenge:** The dataset consists of images of *different* individuals, making it difficult to directly compare facial asymmetry and understand what constitutes "abnormal" for a given person without a baseline. An ideal dataset would have before-and-after stroke images of the same person.
* **Imbalance:** The dataset is imbalanced, with over 2500 non-stroke images and around 1200 stroke images.

## Approach

The core of this project is a **Variational Autoencoder (VAE)** implemented using PyTorch.

* **VAE Selection:** VAEs are chosen for their ability to learn a meaningful latent space where similar data points (faces) are clustered together. This property is intended to make it easier to identify and manipulate latent features related to facial characteristics like symmetry.
* **Latent Space Exploration:** A key part of the project involves exploring the learned latent space through hyperparameter tuning to understand how different dimensions correspond to visual features and whether dimensions controlling facial asymmetry can be isolated.
* **Generation:** By manipulating the latent space, the project aims to generate synthetic images showing facial drooping to visualize stroke symptoms.
* **Stroke Detection (Implicit):** While the primary focus is generation, the VAE's reconstruction error can also be used as an anomaly detection mechanism, where high reconstruction error might indicate facial features significantly different from the learned "normal" face manifold (potentially signaling stroke).

## Project Stages

The project development was broken down into the following key stages:

1.  **Data Preprocessing:** Loading images and labels, handling data imbalance, and preparing images for the VAE.
2.  **Model Architecture Initialization:** Defining the structure of the VAE (Encoder and Decoder).
3.  **Model Training:** Training the VAE using the preprocessed data and a defined loss function.
4.  **Testing and Evaluation:** Evaluating the VAE's performance, particularly its ability to detect stroke cases based on reconstruction error.
5.  **Image Generation:** Using the trained model to generate new images by sampling and interpolating in the latent space.

## Challenges and Solutions Implemented/Planned

**1. Data Variability and Lack of Alignment**

* **Challenge:** The dataset consists of unaligned faces of different people, with varying poses, expressions, lighting, and individual features. Without before-and-after images, understanding typical vs. abnormal asymmetry is difficult. Poor alignment makes it hard for the VAE to learn a consistent face representation.
* **Solution:** Implement facial alignment as a critical preprocessing step. This involves:
    * Detecting facial landmarks (like eyes, nose, mouth corners) using libraries such as `dlib` or `mediapipe`.
    * Applying geometric transformations (affine transformations including scaling, rotation, translation) to align key facial points to a standard, predefined pose. This aims to normalize faces across the dataset.

**2. Data Imbalance**

* **Challenge:** The dataset has significantly more non-stroke images than stroke images. Training directly on this imbalance can lead to a model biased towards the majority class.
* **Solution:** Implement **weighted loss** in the VAE's loss function. This assigns a higher penalty to reconstruction errors and KL divergence for samples from the minority class (stroke) compared to the majority class (non-stroke), encouraging the model to pay more attention to the underrepresented data.

**3. Model Training Effectiveness and Image Generation Quality**

* **Challenge:** Initial training resulted in poor image generation quality (unrecognizable faces), indicating the VAE was not effectively learning the face manifold.
* **Solutions:**
    * **Hyperparameter Tuning:** Conducted a grid search to explore different combinations of learning rate, latent space dimension, and the $\beta$ parameter (weight for KL divergence in the loss function).
    * **Model Architecture Refinement:** Added Batch Normalization layers to the VAE architecture to help stabilize and accelerate training.
    * **Increased Training Epochs:** Increased the number of training epochs to allow the model more time to converge.
    * **Learning Rate Scheduling:** Integrated a learning rate scheduler (e.g., `ReduceLROnPlateau`) into the training process to dynamically adjust the learning rate based on validation performance, aiding convergence.
    * **Visual Monitoring:** Added functionality to periodically save sample reconstructed and generated images during training to visually track learning progress beyond just loss metrics.
    * **Robust Evaluation:** Improved the evaluation process (`evaluate.py`) to correctly load the best model's hyperparameters and weights, display original vs. reconstructed images from the test set, and perform latent space sampling and interpolation.

## Technical Implementation Details

* **Model:** Variational Autoencoder (VAE) with convolutional and transposed convolutional layers, ReLU activation, and Batch Normalization.
* **Loss Function:** Weighted VAE Loss = Weighted Binary Cross-Entropy (Reconstruction Loss) + $\beta$ * Kullback-Leibler Divergence (Regularization Loss).
* **Optimizer:** Adam.
* **Libraries:** PyTorch, NumPy, Scikit-learn, Pillow, OpenCV, Mediapipe, Matplotlib.
* **Preprocessing:** Grayscale conversion, resizing, normalization (0-1), facial alignment (rotation, cropping, resizing), PyTorch Tensors.
* **Data Loading:** Custom PyTorch `Dataset` and `DataLoader` with multiprocessing (`num_workers`).

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy scikit-learn Pillow opencv-python mediapipe matplotlib
    ```
3.  **Download and Organize Data:**
    * Download the dataset from the Kaggle link provided above.
    * Organize the images into directories named `stroke_data` and `noStroke_data` inside a `data` directory at the root of the project.
    ```
    .
    ├── data/
    │   ├── stroke_data/
    │   │   ├── image1.jpg
    │   │   ├── ...
    │   └── noStroke_data/
    │       ├── imageA.jpg
    │       ├── ...
    ├── model.py
    ├── preprocess.py
    ├── train.py
    ├── evaluate.py
    ├── generate.py
    └── README.md
    ```
4.  **Run Preprocessing (implicitly by running train/evaluate):** The `preprocess.py` script defines how data is loaded and processed. You don't run it directly as a main script, but the `train.py`, `evaluate.py`, and `generate.py` scripts import and use its components (`train_loader`, `val_loader`, `test_loader`, `class_weights_tensor`).
5.  **Train the VAE:**
    ```bash
    python train.py
    ```
    This will perform the grid search and save the best model weights and hyperparameters to `./best_vae_stroke_model.pth`. Sample reconstruction and generated images will be saved periodically in `./sample_images`.
6.  **Evaluate the Model:**
    ```bash
    python evaluate.py
    ```
    This script loads the best trained model, evaluates its stroke detection performance on the test set, displays evaluation metrics (Confusion Matrix, F1-score, ROC curve), shows reconstructions of test images, and demonstrates latent space sampling and interpolation.
7.  **Generate Images:**
    ```bash
    python generate.py
    ```
    This script loads the best trained model and generates a single image by sampling a random vector from the latent space, then displays it.

## Results and Future Work

* Initial attempts at image generation yielded poor results, highlighting the challenges posed by the dataset's variability and lack of alignment.
* The implemented improvements, particularly facial alignment and refined training strategies, are expected to significantly improve the quality of generated images and the meaningfulness of the latent space.
* Evaluation metrics provide insight into the model's ability to distinguish stroke vs. non-stroke based on reconstruction error, which serves as an anomaly detection signal.
* **Future Work:** A key next step is to systematically explore the learned latent space of the *aligned* faces to identify specific dimensions or combinations of dimensions that correlate with facial drooping or asymmetry. This could involve techniques like traversing individual latent dimensions or performing latent space arithmetic (e.g., `mean_stroke_faces - mean_non_stroke_faces`) to find a "stroke asymmetry" vector. Manipulating faces along such a vector could allow controlled generation of images mimicking stroke symptoms.

---

This README provides a comprehensive overview of your project. You can save this content as `README.md` in your project's root directory.

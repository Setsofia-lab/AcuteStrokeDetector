# Acute Stroke Detector: Facial Asymmetry Analysis using VAE  

## Project Goal  
This project explores the use of a Variational Autoencoder (VAE) to analyze facial images for signs of acute stroke, focusing on detecting facial asymmetry or "drooping." A secondary goal is to leverage the VAE's generative capabilities to create images mimicking stroke symptoms, which could aid in education or diagnostics.  

## Original Project Proposal  
**Acute stroke detection using images of facial asymmetry:**  
The core task is to identify patterns in facial images that correlate with acute stroke, primarily through the analysis of facial symmetry.  

- **Images Source:**  
    The project utilizes the Kaggle dataset ["Face Images of Acute Stroke and Non-Acute Stroke"](https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke). The dataset contains images of different individuals, requiring the model to learn general patterns of asymmetry rather than comparing a person's face before and after a stroke. The dataset is imbalanced, with more non-stroke images than stroke images.  

- **Model Architecture:**  
    A Variational Autoencoder (VAE) is chosen for its ability to learn a compressed, smooth latent space representation of faces. This facilitates detecting features related to facial drooping and generating new facial images.  

- **Extra Criteria:**  
    Exploration of the latent space through hyperparameter tuning to understand dimensions controlling facial symmetry and generating images mimicking stroke symptoms.  

## Project Approach and Development Phases  

### a. Data Preprocessing  
- Loading image file paths and corresponding stroke/non-stroke labels.  
- Addressing dataset imbalance using weighted loss during training.  
- Splitting data into training, validation, and test sets using stratified sampling.  
- Preprocessing pipeline: grayscale conversion, resizing to 64x64, and pixel normalization to [0, 1].  
- Creating efficient DataLoaders with `num_workers` for parallel data loading.  

### b. Model Architecture Initialization  
- Designing and implementing the VAE model from scratch using PyTorch.  
- Encoder: Convolutional layers; Decoder: Transposed Convolutional layers.  
- Incorporating the Reparameterization Trick for differentiable sampling.  
- Adding Batch Normalization (`nn.BatchNorm2d`) for training stability.  

### c. Model Training  
- Loss function: Reconstruction Loss (Binary Cross-Entropy) + KL Divergence Loss.  
- Applying class weights to mitigate data imbalance bias.  
- Optimizer: Adam.  
- Training loop with gradient clipping to prevent exploding gradients.  
- Validation loop to monitor performance on unseen data.  

### d. Testing and Evaluation of Model Performance  
- Using the VAE as an anomaly detector based on Reconstruction Error.  
- Calculating reconstruction errors for test set images.  
- Determining an optimal threshold for classification (e.g., maximizing F1-score).  
- Reporting metrics: Confusion Matrix, Precision, Recall, F1-Score, Accuracy.  
- Generating and visualizing the ROC curve and calculating AUC.  

### e. Testing and Image Generation  
- Sampling random points from the latent space to generate new facial images.  
- Latent space interpolation to visualize smooth transitions between features.  

## Challenges Encountered and Solutions  

### Initial Challenges  
1. **Poor Model Performance and Unrealistic Generated Images:**  
     - Low evaluation metrics and poor image quality indicated ineffective learning.  
2. **Training Instability:**  
     - Higher learning rates caused numerical instability (e.g., BCE calculation errors).  

### Solutions  
- **Hyperparameter Tuning:**  
    Grid search over key hyperparameters (e.g., `learning_rate`, `latent_dim`, `beta`) to improve performance.  
- **Batch Normalization:**  
    Adding `nn.BatchNorm2d` layers improved stability and allowed higher learning rates.  

## Future Improvements and Next Steps  

### 1. Facial Alignment  
- **Challenge:** Variability in face pose, expression, and alignment.  
- **Solution:** Use libraries like `dlib` or `mediapipe` to detect facial landmarks and align faces to a standard pose.  

### 2. Increased Model Capacity  
- **Challenge:** Current architecture may lack capacity to capture complex features.  
- **Solution:** Increase model depth or width while avoiding overfitting.  

### 3. Learning Rate Scheduling  
- **Challenge:** Fixed learning rate may not be optimal.  
- **Solution:** Use a learning rate scheduler (e.g., `torch.optim.lr_scheduler`) to adjust rates dynamically.  

### 4. Monitoring Training Progress with Generated Samples  
- **Challenge:** Lack of visual feedback during training.  
- **Solution:** Periodically generate and save sample images to monitor progress.  

### 5. Saving Best Hyperparameters  
- **Challenge:** Tracking hyperparameters during grid search.  
- **Solution:** Save both `model.state_dict()` and hyperparameters in a dictionary.  

### 6. Enhanced Evaluation Script  
- **Challenge:** Limited evaluation visualization.  
- **Solution:** Modify the script to display original vs. reconstructed images and reconstruction errors.  

By implementing these improvements, the project aims to enhance the model's robustness, training process, and evaluation tools, moving closer to detecting stroke-related facial asymmetry effectively.  

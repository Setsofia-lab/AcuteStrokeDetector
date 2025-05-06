import torch
import torch.nn as nn
import torch.nn.functional as F

# Input image dimensions
IMG_CHANNELS = 1 
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Latent space dimension 
LATENT_DIM = 256

class VAE(nn.Module):
    def __init__(self, img_channels, img_height, img_width, latent_dim):
        super(VAE, self).__init__()

        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim

        # The encoder takes an image and outputs the parameters (mean and log-variance) of the latent distribution q(z|x).convolutional layers used to extract features.
        self.encoder = nn.Sequential(
            # Input: [batchsize, channel, height of input, width of input]. Example for 64x64 input is  [batch_size, 1, 64, 64]
            # Through out the encoder the channels will increase but the image dimensions reduces into a compressed form
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1), # Output: [batch_size, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            # nn.BatchNorm()
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # The final output from the encoder is a 4x4 feature maps with 256 channels.This vecor will now be flatten to a 1d array.  batch_size, 256 * 4 * 4 = [batch_size, 4096]
            nn.Flatten() 
        )
        # Calculating the size of the flattened layer output is a bit tricky so we calculate manually, so we can pass a dummy tensor through the encoder to get the size dynamically.
        self.flattened_size = self._get_flattened_size()
        # The dense layers map the flattened features to the latent space parameters.
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim) # Output: [batch_size, latent_dim] (mean vector)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim) # Output: [batch_size, latent_dim] (log-variance vector)

        # The decoder takes a sample from the latent space (z) and reconstructs an image. It's essentially the reverse of the encoder, using transposed convolutions for upsampling.
        # Initial dense layer to map latent vector back to a feature map size This size should match the output of the last conv layer *before* flattening in the encoder.
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size) # Output: [batch_size, 4096]

        # Transposed Convolutional layers (for upsampling)
        self.decoder = nn.Sequential(
            # Input is reshaped from [batch_size, 4096] to [batch_size, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # ConvTranspose layer 4: Output image channels (1 for grayscale)The final layer uses Sigmoid to output pixel values between 0 and 1.
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid()
        )
    # Helper function to calculate the size of the flattened layer output dynamically
    def _get_flattened_size(self):
        # Create a dummy input tensor with the expected shape
        dummy_input = torch.randn(1, self.img_channels, self.img_height, self.img_width)
        # Pass it through the encoder layers up to (but not including) the dense layers. Use a temporary sequential model for this calculation
        temp_encoder = nn.Sequential(
            # Copy the layers from the encoder sequential block
            nn.Conv2d(self.img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad(): # Don't track gradients for this calculation
            flattened_output = temp_encoder(dummy_input)
        return flattened_output.size(1) # Return the size of the flattened dimension

    # Reparameterization Trick - This function samples a latent vector z from the distribution defined by mu and logvar.It's done in a way that allows gradients to flow back during training.
    def reparameterize(self, mu, logvar):
        # Calculate standard deviation from log-variance
        # log(sigma^2) = logvar
        # sigma^2 = exp(logvar)
        # sigma = sqrt(exp(logvar)) = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)

        # Sample epsilon from a standard normal distribution (mean 0, variance 1) The shape of epsilon should match the shape of std and mu
        epsilon = torch.randn_like(std) # randn_like creates a tensor of the same shape and device

        # Calculate the latent vector z using the reparameterization trick formula
        # z = mu + sigma * epsilon
        z = mu + epsilon * std
        return z

    #Forward Pass - This defines how data flows through the VAE during training and inference.
    def forward(self, x):
        # Pass the input image through the encoder. The encoder outputs the flattened features
        encoded_features = self.encoder(x)

        # Pass the flattened features through the dense layers to get mean and log-variance
        mu = self.fc_mu(encoded_features)
        logvar = self.fc_logvar(encoded_features)

        # Use the reparameterization trick to sample a latent vector z
        z = self.reparameterize(mu, logvar)

        # Pass the latent vector z through the decoder. First, map the latent vector back to the shape expected by the transposed convolutions
        decoder_input = self.decoder_input(z)
        # Reshape the output to the spatial dimensions expected by the first ConvTranspose layer. The shape should be [batch_size, channels, height, width]
        # The channels and spatial dimensions should match the output of the last conv before flatten in encoder. For 64x64 example, this is [batch_size, 256, 4, 4]
        decoder_input = decoder_input.view(-1, 256, 4, 4) # -1 infers the batch size

        # Pass through the decoder's transposed convolutional layers
        reconstructed_x = self.decoder(decoder_input)

        # Return the reconstructed image, the mean vector, and the log-variance vector. The mean and log-variance are needed to calculate the KL divergence loss.
        return reconstructed_x, mu, logvar


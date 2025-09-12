import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class AudioModel1(nn.Module):
    """
    In this model, we take a 15second audio clip in wav format as the input at 16kHz refresh rate.
    We use this audio format and first we add to layers to the input layer with the layers being
        1. Original Audio Clip
        2. the Modulus of the Audio Clip
        3. The sign of the Audio Clip ie. -1,0,1

    We then pass this 3 channels into a convolutional encoding block.
    In the end mimicing a VAE we have a latent layer with 128 channels, that is .

    There are 2 potential outputs of this model. 
        1. One will be a classification output. The output dimension of this layer is 15.
        2. The second will be a reconstruction output. The output dimension of this layer is (1, 240000)
            which is the same as the input dimension.
    Args:
        input_channels (int): Number of input channels. Default is 1 for mono audio.
        num_classes (int): Number of output classes for classification. Default is 15.

    """

    def __init__(self, input_channels=1, num_classes=15, latent_dim=128):
        super(AudioModel1, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        conv1_channels = input_channels * 3  # Original, Modulus, Sign channels

        # Activation function
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2)

        # --- Convolutional Layers ---
        # These layers are designed to extract features from the input audio waveform.

        self.conv1 = nn.Conv1d(in_channels=conv1_channels,
                               out_channels=32,
                               kernel_size=8,
                               stride=4,
                               padding=2)  # (B, 3, 240000) -> (B, 32, 60000)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               stride=3,
                               padding=1)  # (B, 32, 60000) -> (B, 64, 20000)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=4,
                               stride=2
                               )  # (B, 64, 20000) -> (B, 128, 10000)

        self.encoder_transformer = TransformerBlock(
            dim=128, num_heads=4, ff_mult=4, dropout=0.1)
        self.bn3 = nn.BatchNorm1d(128)

        # --- Latent Layer ---
        # This layer represents the compressed representation of the input audio.
        self.fc_mu = nn.Linear(128 * 10000, latent_dim)
        self.fc_log_var = nn.Linear(128 * 10000, latent_dim)

        # --- Fully Connected Layers for Classification ---
        # These layers will perform the classification based on the extracted features.
        self.fc1 = nn.Linear(latent_dim, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # --- Fully Connected Layers for Reconstruction ---
        # These layers will reconstruct the input audio from the latent representation.
        self.fc3 = nn.Linear(latent_dim, 128 * 5000)
        self.decoder_transformer = TransformerBlock(
            dim=128, num_heads=4, ff_mult=4, dropout=0.1)
        self.deconv1 = nn.ConvTranspose1d(in_channels=128,
                                          out_channels=64,
                                          kernel_size=4,
                                          stride=2,
                                          # (B, 128, 5000) -> (B, 64, 10000)
                                          padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(in_channels=64,
                                          out_channels=32,
                                          kernel_size=6,
                                          stride=4,
                                          # (B, 64, 10000) -> (B, 32, 40000)
                                          padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.deconv3 = nn.ConvTranspose1d(in_channels=32,
                                          out_channels=2,
                                          kernel_size=8,
                                          stride=6,
                                          # (B, 32, 40000) -> (B, 2, 240000)
                                          padding=1)

    def encoder(self, x):
        """
        Encoder part of the model that processes the input audio waveform.
        """
        # Prepare input with 3 channels: original, modulus, sign
        x_modulus = torch.abs(x)
        x_sign = torch.sign(x)
        x = torch.cat([x, x_modulus, x_sign], dim=1)  # (B, 3, samples)
        # Pass through convolutional layers
        x = self.leaky_relu(self.bn1(self.conv1(x)))  # (B, 32, 60000)
        x = self.leaky_relu(self.bn2(self.conv2(x)))  # (B, 64, 20000)
        x = self.leaky_relu(self.bn3(self.conv3(x)))  # (B, 128, 10000)
        x = x.permute(0, 2, 1)  # (B, 10000, 128) for transformer
        x = self.encoder_transformer(x)
        x = x.permute(0, 2, 1)  # (B, 128, 10000)
        x = x.contiguous().view(x.size(0), -1)  # Flatten for FC layer
        # Latent representation
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, latent):
        """
        Decoder part of the model that reconstructs the audio waveform from the latent representation.
        returns: reconstructed audio waveform
        """
        x = self.leaky_relu(self.fc3(latent))
        x = x.view(x.size(0), 128, 5000)  # (B, 128, 5000)
        x = x.permute(0, 2, 1)  # (B, 5000, 128) for transformer
        x = self.decoder_transformer(x)
        x = x.permute(0, 2, 1)  # (B, 128, 5000)
        x = self.leaky_relu(self.bn4(self.deconv1(x)))  # (B, 64, 10000)
        x = self.leaky_relu(self.bn5(self.deconv2(x)))  # (B, 32, 40000)
        x = self.deconv3(x)  # (B, 2, 240000)

        return x

    def classifier(self, latent):
        """
        Classifier part of the model that predicts the class from the latent representation.
        returns: class logits
        """
        x = self.leaky_relu(self.batchnorm1(self.fc1(latent)))
        x = self.fc2(x)
        return x

    def forward(self, x):
        """
        Forward pass with reparameterization.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        class_logits = self.classifier(z)
        reconstructed = self.decoder(z)
        return class_logits, reconstructed, mu, log_var

    def loss_function(self, x, class_targets, alpha=0.5, beta=0.001):
        """
        Combined loss function with KL divergence.

        Args:
            x (torch.Tensor): Input audio waveform.
            class_targets (torch.Tensor): True class labels.
            alpha (float): Weight for classification vs reconstruction loss.
            beta (float): Weight for KL divergence loss.

        Returns:
            dict: Dictionary of losses.
        """
        class_logits, reconstructed, mu, log_var = self.forward(x)

        # Classification loss
        classification_loss = F.cross_entropy(class_logits, class_targets)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # KL divergence loss :cite[1]:cite[5]
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.size(0)  # Average over batch

        # Total loss
        total_loss = alpha * classification_loss + \
            (1 - alpha) * reconstruction_loss + beta * kl_loss

        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, MelSpectrogram, InverseMelScale, GriffinLim


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


class AudioEncoder1(nn.Module):
    def __init__(self, input_channels=1, num_classes=15, latent_dim=128, n_fft=1024, hop_length=256, n_mels=128):
        super(AudioEncoder1, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        conv1_channels = input_channels * 3  # Original, Modulus, Sign channels

        # Activation function
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2)

        # --- Convolutional Layers ---
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

        # --- Frequency Domain Processing ---
        # Add frequency encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # Audio transforms for frequency processing
        self.to_mel = MelSpectrogram(
            n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.inverse_mel = InverseMelScale(n_stft=n_fft//2 + 1, n_mels=n_mels)
        self.griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length)

        # Calculate dimensions after convolutions
        time_features = 128 * 10000  # As in your original code
        freq_features = 128 * (n_mels // 4) * ((240000 // hop_length + 1) // 4)

        # Update latent layer to handle combined features
        self.fc_mu = nn.Linear(time_features + freq_features, latent_dim)
        self.fc_log_var = nn.Linear(time_features + freq_features, latent_dim)

        # --- Fully Connected Layers for Classification ---
        self.fc1 = nn.Linear(latent_dim, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # --- Reconstruction Decoders ---
        # Modulus decoder (replaces the original waveform decoder)
        self.modulus_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 5000),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 5000)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=6, padding=1),
            nn.ReLU()  # Modulus is always non-negative
        )

        # Frequency decoder (for sign extraction)
        self.freq_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (n_mels//4)
                      * ((240000//hop_length+1)//4)),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, n_mels//4, (240000//hop_length+1)//4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()  # Magnitude spectrograms are non-negative
        )

    def encode(self, x):
        """
        Encoder part of the model that processes both time and frequency domains.
        """
        # Time-domain encoding
        x_modulus = torch.abs(x)
        x_sign = torch.sign(x)
        x_time = torch.cat([x, x_modulus, x_sign], dim=1)  # (B, 3, samples)

        # Pass through convolutional layers
        x_time = self.leaky_relu(
            self.bn1(self.conv1(x_time)))  # (B, 32, 60000)
        x_time = self.leaky_relu(
            self.bn2(self.conv2(x_time)))  # (B, 64, 20000)
        x_time = self.leaky_relu(
            self.bn3(self.conv3(x_time)))  # (B, 128, 10000)
        x_time = x_time.permute(0, 2, 1)  # (B, 10000, 128) for transformer
        x_time = self.encoder_transformer(x_time)
        x_time = x_time.permute(0, 2, 1)  # (B, 128, 10000)
        time_features = x_time.contiguous().view(x_time.size(0), -1)  # Flatten

        # Frequency-domain encoding
        spectrogram = self.to_mel(x).unsqueeze(1)  # Add channel dimension
        freq_features = self.freq_encoder(spectrogram)
        freq_features = freq_features.contiguous().view(freq_features.size(0), -1)

        # Combine features
        combined = torch.cat([time_features, freq_features], dim=1)

        # Latent representation
        mu = self.fc_mu(combined)
        log_var = self.fc_log_var(combined)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, original_audio=None, teacher_forcing=True):
        """
        Decoder part that reconstructs modulus and extracts sign from frequency domain.
        """
        # Reconstruct modulus from time domain
        modulus_prediction = self.modulus_decoder(z)

        # Extract sign using either teacher forcing or frequency reconstruction
        if teacher_forcing and original_audio is not None and self.training:
            # Use true sign during training (teacher forcing)
            sign_prediction = torch.sign(original_audio)
        else:
            # Reconstruct frequency domain and get waveform for sign extraction
            freq_reconstruction = self.freq_decoder(z).squeeze(1)
            linear_spec = self.inverse_mel(freq_reconstruction)
            waveform_reconstruction = self.griffin_lim(linear_spec)

            # Extract sign from frequency-based reconstruction
            epsilon = 1e-8
            normalized_waveform = waveform_reconstruction / \
                (torch.abs(waveform_reconstruction) + epsilon)
            sign_prediction = torch.sign(normalized_waveform)

        # Combine modulus prediction with sign prediction
        final_reconstruction = modulus_prediction * sign_prediction

        return final_reconstruction, modulus_prediction, sign_prediction

    def classifier(self, latent):
        """
        Classifier part of the model that predicts the class from the latent representation.
        returns: class logits
        """
        x = self.leaky_relu(self.batchnorm1(self.fc1(latent)))
        x = self.fc2(x)
        return x

    def forward(self, x, teacher_forcing=True):
        """
        Forward pass with reparameterization.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        class_logits = self.classifier(z)
        final_recon, modulus_recon, sign_recon = self.decode(
            z, x, teacher_forcing)
        return class_logits, final_recon, modulus_recon, sign_recon, mu, log_var

    def loss_function(self, x, class_targets, alpha=0.5, beta=0.001, gamma=0.7):
        """
        Combined loss function with focus on modulus and sign reconstruction.
        """
        class_logits, final_recon, modulus_recon, sign_recon, mu, log_var = self.forward(
            x, teacher_forcing=True)

        # Get true modulus and sign
        x_modulus = torch.abs(x)
        x_sign = torch.sign(x)

        # Classification loss
        classification_loss = F.cross_entropy(class_logits, class_targets)

        # Modulus reconstruction loss
        modulus_recon_loss = F.mse_loss(modulus_recon, x_modulus)

        # Sign loss (using cross-entropy for discrete classification)
        # Convert sign to class labels: -1 -> 0, 0 -> 1, 1 -> 2
        sign_classes = x_sign.squeeze(1) + 1  # Convert to 0, 1, 2
        sign_classes = sign_classes.long()

        # Convert the continuous sign prediction to class probabilities
        sign_pred_probs = torch.stack([
            -sign_recon.squeeze(1),  # Probability of -1 (inverse relationship)
            torch.ones_like(sign_recon.squeeze(1)),  # Constant for zero
            sign_recon.squeeze(1)  # Probability of +1
        ], dim=-1)

        sign_recon_loss = F.cross_entropy(sign_pred_probs, sign_classes)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.size(0)

        # Total loss - focus only on modulus and sign reconstruction
        total_loss = (
            alpha * classification_loss +
            (1 - alpha) * (gamma * modulus_recon_loss + (1 - gamma) * sign_recon_loss) +
            beta * kl_loss
        )

        # Calculate sign accuracy for monitoring
        predicted_sign = torch.argmax(
            sign_pred_probs, dim=-1) - 1  # Convert back to -1, 0, 1
        sign_accuracy = (predicted_sign == x_sign.squeeze(1)).float().mean()

        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'modulus_recon_loss': modulus_recon_loss,
            'sign_recon_loss': sign_recon_loss,
            'sign_accuracy': sign_accuracy,
            'kl_loss': kl_loss
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        cross1, _ = self.attention(query=x1, key=x2, value=x2)
        return self.norm1(x1 + cross1)


class AudioEncoder1(nn.Module):
    def __init__(self, input_channels=1, num_classes=15, latent_dim=500):
        super(AudioEncoder1, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        conv1_channels = input_channels * 3  # Original, Modulus, Sign channels

        # Activation function
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.2)

        # ---- Convolutional Layers ----

        # ---- Waveform convolutions ----
        # Parallel conv 1
        self.conv1_1 = nn.Conv1d(in_channels=conv1_channels,
                                 out_channels=32,
                                 kernel_size=8,
                                 stride=4,
                                 padding=2)  # (B, 3, 240000) -> (B, 32, 60000)
        self.bn32 = nn.BatchNorm1d(32)
        self.conv1_2 = nn.Conv1d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=5,
                                 stride=3,
                                 padding=1)  # (B, 32, 60000) -> (B, 64, 20000)
        self.bn128 = nn.BatchNorm1d(128)

        # Parallel conv 2
        self.conv2_1 = nn.Conv1d(in_channels=conv1_channels,
                                 out_channels=32,
                                 kernel_size=5,
                                 stride=3,
                                 padding=1)  # (B, 3, 240000) -> (B, 32, 80000)

        self.conv2_2 = nn.Conv1d(in_channels=32,
                                 out_channels=64,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)  # (B, 32, 80000) -> (B, 64, 40000)
        self.bn64 = nn.BatchNorm1d(64)

        self.conv2_3 = nn.Conv1d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)  # (B, 64, 40000) -> (B, 128, 20000)

        # Final conv
        self.final_conv_1 = nn.Conv1d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1)  # (B, 128+128, 20000) -> (B, 128, 10000)
        self.final_bn128 = nn.BatchNorm1d(128)
        self.final_conv2 = nn.Conv1d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1)  # (B, 128, 10000) -> (B, 128, 5000)
        self.final_conv3 = nn.Conv1d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=7,
                                     stride=5,
                                     padding=1)  # (B, 128, 5000) -> (B, 128, 1000)

        # ---- Spectogram convolutions ----
        # Frequency conv
        self.n_fft_f = 4000
        self.hop_length_f = 1200
        self.window_size_f = 2000

        self.to_spec_f = Spectrogram(
            n_fft=self.n_fft_f,
            hop_length=self.hop_length_f,
            win_length=self.window_size_f,
            power=None,  # None returns complex values
            center=True
        )  # (B, 1, 240000) -> (B, 1, 2001, 201)

        # Time conv
        self.n_fft_t = 1000
        self.hop_length_t = 300
        self.window_size_t = 1000

        self.to_spec_t = Spectrogram(
            n_fft=self.n_fft_t,
            hop_length=self.hop_length_t,
            win_length=self.window_size_t,
            power=None,  # None returns complex values
            center=True
        )  # (B, 1, 240000) -> (B, 1, 501, 801)

        self.freq_conv1 = nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=(5, 3),
                                    stride=(4, 1),
                                    # (B, 1, 2001, 201) -> (B, 32, 501, 201)
                                    padding=(2, 1)
                                    )
        self.freq_bn32 = nn.BatchNorm2d(32)

        self.time_conv1 = nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=(3, 5),
                                    stride=(1, 4),
                                    # (B, 1, 501, 801) -> (B, 32, 501, 201)
                                    padding=(1, 2)
                                    )
        self.time_bn32 = nn.BatchNorm2d(32)

        # Combined conv
        self.combined_conv1 = nn.Conv2d(in_channels=64,
                                        out_channels=16,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)  # (B, 32+32, 501, 201) -> (B, 16, 501, 201)

        self.combined_bn16 = nn.BatchNorm2d(16)

        self.combined_conv2 = nn.Conv2d(in_channels=16,
                                        out_channels=1,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)  # (B, 16, 501, 201) -> (B, 1, 251, 101)

        self.combined_1dconv = nn.Conv1d(in_channels=251,
                                         out_channels=128,
                                         kernel_size=2,
                                         stride=1,
                                         padding=0)  # (B, 251, 101) -> (B, 128, 100)

        # ---- Fully Connected Layers for Latent Space ----
        self.cross_attention = CrossAttentionBlock(dim=128, num_heads=16)
        self.fully_connected = nn.Linear(128 * 100, latent_dim * 2)

        # Update latent layer to handle combined features
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim * 2, latent_dim)

        # ---- Decoders ----
        # --- Fully Connected Layers for Classification ---
        self.fc1 = nn.Linear(latent_dim, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # --- Reconstruction Decoders ---
        # Balanced spectogram parameters
        # n_ftt=2000, hop_length=480, win_length=1500
        self.n_fft_recon = 2000
        self.hop_length_recon = 480
        self.win_length_recon = 1500

        # To convert original waveform to spectrogram for training
        self.recon_to_spec = Spectrogram(
            n_fft=self.n_fft_recon,
            hop_length=self.hop_length_recon,
            win_length=self.win_length_recon,
            power=None,  # None returns complex values
            center=True
        )  # (B, 1, 240000) -> (B, 1, 1001, 501)

        # Inverse Reconstructed spectrogram back to waveform
        self.reconspec_to_waveform = InverseSpectrogram(
            n_fft=self.n_fft_recon,
            hop_length=self.hop_length_recon,
            win_length=self.win_length_recon,
            center=True,
        )

        # ConvTranspose to reconstruct spectrogram
        self.fully_connected_dec = nn.Linear(latent_dim, 32 * 100)
        self.convT_1 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=7,
                                          stride=5,
                                          padding=1)  # (B, 32, 10, 10) -> (B, 16, 50, 50)
        self.bn32_dec = nn.BatchNorm2d(32)
        self.convT_2 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=16,
                                          kernel_size=6,
                                          stride=4,
                                          padding=1)  # (B, 32, 50, 50) -> (B, 16, 200, 200)
        self.bn16_dec = nn.BatchNorm2d(16)
        self.convT_3 = nn.ConvTranspose2d(in_channels=16,
                                          out_channels=8,
                                          kernel_size=7,
                                          stride=5,
                                          padding=1)  # (B, 16, 200, 200) -> (B, 8, 1000, 1000)
        self.dec_conv1 = nn.Conv2d(in_channels=8,
                                   out_channels=1,
                                   kernel_size=4,
                                   stride=(1, 2),
                                   padding=2)  # (B, 8, 1000, 1000) -> (B, 1, 1001, 501)

    def waveform_branch(self, x):
        """
        Processes the waveform through convolutional layers and a transformer.
        """
        x_modulus = torch.abs(x)
        x_sign = torch.sign(x)
        x_time = torch.cat([x, x_modulus, x_sign], dim=1)  # (B, 3, samples)

        # Pass through convolutional layers
        x1 = self.leaky_relu(self.bn32(self.conv1_1(x_time)))  # (B, 32, 60000)
        x1 = self.leaky_relu(self.bn128(self.conv1_2(x1)))  # (B, 128, 20000)

        x2 = self.leaky_relu(self.bn32(self.conv2_1(x_time)))  # (B, 32, 80000)
        x2 = self.leaky_relu(self.bn64(self.conv2_2(x2)))  # (B, 64, 40000)
        x2 = self.leaky_relu(self.bn128(self.conv2_3(x2)))  # (B, 128, 20000)

        # Concatenate parallel conv outputs
        x_combined = torch.cat([x1, x2], dim=1)  # (B, 128+128=256, 20000)

        # Final conv layers
        x_combined = self.leaky_relu(
            self.final_bn128(self.final_conv_1(x_combined)))  # (B, 128, 10000)
        x_combined = self.leaky_relu(
            self.final_bn128(self.final_conv2(x_combined)))  # (B, 128, 5000)
        x_combined = self.leaky_relu(
            self.final_conv3(x_combined))  # (B, 128, 1000)

        return x_combined

    def spectrogram_branch(self, x):
        """
        Processes the spectrogram through convolutional layers.
        """
        # Frequency-domain spectrogram
        spectrogram_f = self.to_spec_f(x).unsqueeze(1)  # Add channel dimension
        x_f = self.leaky_relu(self.freq_bn32(
            self.freq_conv1(spectrogram_f)))  # (B, 32, 501, 201)

        # Time-domain spectrogram
        spectrogram_t = self.to_spec_t(x).unsqueeze(1)  # Add channel dimension
        x_t = self.leaky_relu(self.time_bn32(
            self.time_conv1(spectrogram_t)))  # (B, 32, 501, 201)

        # Combine frequency and time features
        x_combined = torch.cat([x_f, x_t], dim=1)  # (B, 32+32=64, 501, 201)
        x_combined = self.leaky_relu(
            self.combined_bn16(self.combined_conv1(x_combined)))  # (B, 16, 501, 201)
        x_combined = self.leaky_relu(
            self.freq_conv2(x_combined))  # (B, 1, 251, 101)
        x_combined = x_combined.permute(0, 2, 3, 1).flatten(2)  # (B, 251, 101)
        x_combined = self.leaky_relu(
            self.combined_1dconv(x_combined))  # (B, 128, 100)
        return x_combined

    def combine_branches(self, x):
        """
        Combines the outputs of the waveform and spectrogram branches using cross-attention.
        """
        x_waveform = self.waveform_branch(x)  # (B, 128, 1000)
        x_spectrogram = self.spectrogram_branch(x)  # (B, 128, 100)

        # Prepare for cross-attention
        x_waveform = x_waveform.permute(0, 2, 1)  # (B, 1000, 128)
        x_spectrogram = x_spectrogram.permute(0, 2, 1)  # (B, 100, 128)
        x_waveform = self.cross_attention(
            x_spectrogram, x_waveform)  # (B, 100, 128)
        x_waveform = x_waveform.permute(0, 2, 1)  # (B, 128, 100)
        return x_waveform

    def encode(self, x):
        """
        Encoder part that combines waveform and spectrogram branches and outputs latent variables.
        """
        x_combined = self.combine_branches(x)  # (B, 128, 100)
        x_flat = x_combined.flatten(1)  # (B, 128*100)

        # Pass through fully connected layer
        h = self.leaky_relu(self.fully_connected(x_flat))  # (B, latent_dim*2)

        # Get mean and log variance for reparameterization
        mu = self.fc_mu(h)  # (B, latent_dim)
        log_var = self.fc_log_var(h)  # (B, latent_dim)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classifier(self, latent):
        """
        Classifier part of the model that predicts the class from the latent representation.
        returns: class logits
        """
        x = self.leaky_relu(self.batchnorm1(self.fc1(latent)))
        x = self.fc2(x)
        return x

    def reconstruct_spec(self, latent):
        """
        Decoder part that reconstructs the spectrogram from the latent representation.
        """
        # Fully connected layer to expand latent space
        x = self.leaky_relu(self.fully_connected_dec(latent))  # (B, 32*100)
        x = x.view(-1, 32, 10, 10)  # Reshape to (B, 32, 10, 10)

        # ConvTranspose layers to reconstruct spectrogram
        x = self.leaky_relu(self.bn32_dec(self.convT_1(x)))  # (B, 32, 50, 50)
        x = self.leaky_relu(self.bn16_dec(
            self.convT_2(x)))  # (B, 16, 200, 200)
        x = self.leaky_relu(self.convT_3(x))  # (B, 8, 1000, 1000)
        x = self.dec_conv1(x)  # (B, 1, 1001, 501)

        return x

    def reconstruct_waveform(self, spec_recon):
        """
        Reconstructs the waveform from the reconstructed spectrogram.
        """
        waveform_recon = self.reconspec_to_waveform(
            spec_recon, length=240000)  # (B, 1, 240000)
        return waveform_recon

    def forward(self, x):
        """
        Forward pass through the model.
        """
        mu, log_var = self.encode(x)  # Encode input to latent space
        z = self.reparameterize(mu, log_var)  # Sample from latent space

        class_logits = self.classifier(z)  # Classify from latent space

        # Reconstruct spectrogram from latent space
        spec_recon = self.reconstruct_spec(z)

        return class_logits, spec_recon, mu, log_var

    def loss_function(self, class_logits, spec_recon, mu, log_var, x, class_labels, alpha=1.0, beta=0.1):
        """
        Computes the combined loss: classification loss + reconstruction loss + KL divergence.
        """
        # Classification loss
        class_loss = F.cross_entropy(class_logits, class_labels)

        # Reconstruction loss
        spec_target = self.recon_to_spec(x).unsqueeze(1)  # (B, 1, 1001, 501)
        recon_loss = F.mse_loss(spec_recon, spec_target)

        # KL Divergence
        kl_divergence = -0.5 * \
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Normalize by batch size and latent dim
        kl_divergence /= x.size(0) * self.latent_dim

        # Total loss
        total_loss = class_loss + alpha * recon_loss + beta * kl_divergence

        return total_loss, class_loss, recon_loss, kl_divergence

    def get_class(self, x):
        """
        Get class predictions from input x.
        """
        mu, log_var = self.encode(x)  # Encode input to latent space
        z = self.reparameterize(mu, log_var)  # Sample from latent space
        class_logits = self.classifier(z)  # Classify from latent space
        class_probs = self.softmax(class_logits)
        return class_probs

    def get_reconstruction(self, x):
        """
        Get reconstructed waveform from input x.
        """
        mu, log_var = self.encode(x)  # Encode input to latent space
        z = self.reparameterize(mu, log_var)  # Sample from latent space
        spec_recon = self.reconstruct_spec(z)  # Reconstruct spectrogram
        waveform_recon = self.reconstruct_waveform(
            spec_recon)  # Reconstruct waveform
        return waveform_recon

# Example usage:
# model = AudioEncoder1(input_channels=1, num_classes=15, latent_dim=500)
# audio_input = torch.randn(8, 1, 240000)  # Batch of 8 audio samples
# class_logits, spec_recon, mu, log_var = model(audio_input)
# class_labels = torch.randint(0, 15, (8,))  # Random class labels for the batch
# loss, class_loss, recon_loss, kl_div = model.loss_function(
#     class_logits, spec_recon, mu, log_var, audio_input, class_labels)
# print(f"Total Loss: {loss.item()}, Class Loss: {class_loss.item()}, Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item()}")

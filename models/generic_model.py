import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericAudioModel(nn.Module):
    """
    A generic audio classification model using a simple CNN architecture.
    This model can be used as a baseline or a template for more complex models.

    Args:
        input_channels (int): Number of input channels (e.g., 1 for mono, 2 for stereo).
                              For spectrograms, this would typically be 1.
        num_classes (int): Number of classes for classification.
    """

    def __init__(self, input_channels=1, num_classes=10):
        super(GenericAudioModel, self).__init__()

        # --- Convolutional Layers ---
        # These layers are designed to extract features from the input audio spectrogram.
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # --- Fully Connected Layers ---
        # These layers will perform the classification based on the extracted features.
        # The input size to fc1 depends on the output of the conv layers and the input spectrogram size.
        # This will need to be calculated or determined dynamically.
        # For now, we'll use a placeholder value that you might need to adjust.
        # Example calculation: If input is (1, 128, 128) -> after 3 pools -> (64, 16, 16)
        # Placeholder, adjust this value
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of spectrograms).
                              Shape: (batch_size, channels, height, width)

        Returns:
            torch.Tensor: The output logits from the model.
        """
        # Pass through convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the output for the fully connected layers
        # The view call reshapes the tensor. -1 tells PyTorch to infer the batch size.
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits

        return x

    def calculate_fc1_input_size(self, sample_input_shape):
        """
        Helper function to dynamically calculate the input size for the first fully connected layer.
        Call this before initializing the model if your input dimensions are variable.

        Args:
            sample_input_shape (tuple): The shape of a sample input tensor,
                                        e.g., (1, 1, 128, 128) for a single mono spectrogram.
        """
        with torch.no_grad():
            dummy_input = torch.randn(sample_input_shape)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            # numel gives total elements, divide by batch size
            flattened_size = x.numel() // x.shape[0]
            print(
                f"Calculated input size for the first fully connected layer: {flattened_size}")
            # Now, you would re-initialize fc1 with this size.
            # For simplicity in this example, we hardcode it, but in a real-world scenario,
            # you would pass this value to the __init__ method.
            # self.fc1 = nn.Linear(flattened_size, 128)
        return flattened_size


# --- Example Usage ---
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It demonstrates how to initialize and use the model.

    # --- Configuration ---
    NUM_CLASSES = 10  # Example: 10 different audio classes
    INPUT_CHANNELS = 1  # Mono audio
    SAMPLE_HEIGHT = 128  # Example height of a spectrogram
    SAMPLE_WIDTH = 128  # Example width of a spectrogram
    BATCH_SIZE = 32

    # --- Model Initialization ---
    model = GenericAudioModel(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    print("Model Architecture:")
    print(model)

    # --- Dynamic FC Layer Sizing (Optional but Recommended) ---
    # Create a dummy input tensor to calculate the flattened size
    sample_input_shape = (1, INPUT_CHANNELS, SAMPLE_HEIGHT, SAMPLE_WIDTH)
    fc1_input_size = model.calculate_fc1_input_size(sample_input_shape)

    # Re-initialize the fc1 layer with the correct size
    model.fc1 = nn.Linear(fc1_input_size, 128)
    print(
        f"\nModel's fc1 layer has been updated for the input size {fc1_input_size}.")

    # --- Forward Pass Example ---
    # Create a dummy batch of data
    dummy_batch = torch.randn(
        BATCH_SIZE, INPUT_CHANNELS, SAMPLE_HEIGHT, SAMPLE_WIDTH)

    # Get model output (logits)
    logits = model(dummy_batch)

    # Should be (BATCH_SIZE, NUM_CLASSES)
    print(f"\nShape of the model output (logits): {logits.shape}")
    print("Example usage complete.")

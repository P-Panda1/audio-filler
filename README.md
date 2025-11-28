# Audio Filler - Music Genre VAE

A deep learning project for music genre classification and audio reconstruction using a Variational Autoencoder (VAE) architecture. This project combines spectrogram-based feature extraction with a dual-task learning approach to simultaneously classify music genres and reconstruct audio waveforms.

## Overview

This project implements an encoder-decoder VAE model that:
- **Classifies** music into 15 different genres (blues, classical, country, folk, hiphop, jazz, lofi, metal, pop, rnb, rock-garage, rock-goth, rock-industrial, rock-krautrock, rock-punk)
- **Reconstructs** audio from learned latent representations
- Uses **spectrogram transformations** for efficient audio processing
- Supports **multiple model configurations** for experimentation

## Architecture

The model consists of four main components:

1. **SpectrogramBlock**: Converts raw audio waveforms to frequency-time spectrograms
2. **Encoder**: Multi-branch convolutional network that processes spectrograms and outputs latent space parameters (μ, σ)
3. **Decoder**: Transposed convolutional network that reconstructs spectrograms and classifies genres from latent codes
4. **InvSpecBlock**: Inverse spectrogram transformation that converts reconstructed spectrograms back to audio waveforms

### Model Variants

The project supports multiple configuration variants:
- `default`: Standard encoder-decoder configuration
- `large_model`: Larger encoder and decoder
- `large_encoder`: Large encoder with standard decoder
- `large_decoder`: Standard encoder with large decoder

## Features

- 🎵 **Multi-genre Classification**: Classifies music into 15 distinct genres
- 🔄 **Audio Reconstruction**: Reconstructs audio from compressed latent representations
- 📊 **MLflow Integration**: Experiment tracking and model versioning
- ⚙️ **Configurable Architecture**: YAML-based configuration for easy experimentation
- 🚀 **GPU/MPS Support**: Automatic device detection (CUDA, MPS, CPU)
- 📈 **Comprehensive Logging**: CSV logs for batch-level and performance metrics
- ☁️ **GCS Integration**: Optional support for training on Google Cloud Storage data

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- libsndfile (for audio I/O)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libsndfile1 build-essential

# macOS
brew install ffmpeg libsndfile
```

### Python Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd audio-filler
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
audio-filler/
├── configs/                 # YAML configuration files
│   ├── encoder_1.yaml      # Standard encoder config
│   ├── encoder_2.yaml      # Large encoder config
│   ├── decoder_1.yaml      # Standard decoder config
│   ├── decoder_2.yaml      # Large decoder config
│   ├── spectrogram.yaml    # Spectrogram transformation config
│   └── invspec.yaml        # Inverse spectrogram config
├── src/
│   ├── blocks/             # Reusable neural network blocks
│   │   ├── ConvBlock.py
│   │   ├── ConvTransposeBlock.py
│   │   ├── SpectrogramBlock.py
│   │   └── InvSpecBlock.py
│   ├── models/
│   │   ├── encoders/       # Encoder architectures
│   │   ├── decoders/       # Decoder architectures
│   │   └── combined/       # Combined encoder-decoder model
│   └── train/              # Training scripts
│       ├── train.py        # Main training script
│       ├── train_function.py
│       └── mlflow_runner.py
├── utils/
│   ├── data_loader.py      # Dataset and data loading utilities
│   ├── load_all_configs.py # Configuration loader
│   └── mp3_to_wav.py       # Audio format conversion
├── tools/
│   └── train_large_model.py  # Standalone training script for large models
├── notebooks/              # Jupyter notebooks for experimentation
├── logs/                   # Training logs and checkpoints
└── requirements.txt        # Python dependencies
```

## Usage

### Data Preparation

Organize your music dataset in the following structure:
```
~/data/project/music/
├── blues/
│   ├── song1.mp3
│   └── song2.mp3
├── classical/
│   └── ...
└── ...
```

The dataset will be automatically processed into fixed-length clips (default: 15 seconds) with configurable stride.

### Basic Training

Train a model with default configuration:

```python
from src.models.combined.encoder_decoder import EncoderDecoderModel
from utils.data_loader import MusicGenreDataset
from torch.utils.data import DataLoader
from src.train.mlflow_runner import run_experiment
from utils.load_all_configs import load_all_configs
import torch

# Load configuration
configs = load_all_configs("default")

# Setup device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load dataset
dataset = MusicGenreDataset("~/data/project/music")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = EncoderDecoderModel(configs).to(device)

# Run training with MLflow tracking
run_experiment(
    model=model,
    dataloader=dataloader,
    num_epochs=50,
    lr=1e-4,
    beta_kl=0.001,
    recon_weight=1.0,
    class_weight=0.5,
    device=device,
    experiment_name="MusicGenreVAE_default"
)
```

### Training Large Models

For training large models (especially on cloud infrastructure), use the standalone script:

```bash
python tools/train_large_model.py \
    --gcs-bucket gs://your-bucket \
    --epochs 50 \
    --batch-size 32 \
    --val-split 0.1 \
    --create-archive
```

See `STEPS_RUNPOD.md` for detailed instructions on training on RunPod or similar cloud platforms.

### Training Parameters

- `num_epochs`: Number of training epochs
- `lr`: Learning rate (default: 1e-4)
- `beta_kl`: KL divergence weight in loss function (default: 0.001)
- `recon_weight`: Weight for reconstruction loss (default: 1.0)
- `class_weight`: Weight for classification loss (default: 0.5)
- `batch_size`: Batch size for training
- `device`: Device to use ("cuda", "mps", or "cpu")

### Loss Function

The model optimizes a combined loss function:

```
Total Loss = recon_weight × Reconstruction Loss + 
             beta_kl × KL Divergence Loss + 
             class_weight × Classification Loss
```

- **Reconstruction Loss**: MSE between original and reconstructed spectrograms
- **KL Divergence Loss**: Regularization term for the latent space
- **Classification Loss**: Cross-entropy loss for genre classification

## Configuration

Model architectures are defined in YAML files under `configs/`. Each configuration specifies:
- Layer types and parameters
- Channel dimensions
- Kernel sizes and strides
- Activation functions
- Batch normalization settings

Modify these files to experiment with different architectures. The configuration system supports:
- Multi-branch parallel processing
- Residual connections
- Custom layer groupings (layer1, squish_freq, squish_time, merge_node, secondarylayer)

## Monitoring Training

### MLflow UI

If using MLflow, start the tracking server:

```bash
mlflow ui
```

Then open `http://localhost:5000` to view experiments, metrics, and model artifacts.

### CSV Logs

Training generates two log files:
- `logs/batch_log.csv`: Per-batch metrics (loss components, accuracy)
- `logs/performance_log.csv`: Periodic performance summaries

### Model Checkpoints

Best models (by classification accuracy) are automatically saved to `logs/best_model_epoch{N}.pt`.

## Inference

After training, load and use the model:

```python
import torch
from src.models.combined.encoder_decoder import EncoderDecoderModel
from utils.load_all_configs import load_all_configs

# Load configuration and model
configs = load_all_configs("default")
model = EncoderDecoderModel(configs)
model.load_state_dict(torch.load("logs/best_model_epoch10.pt"))
model.eval()

# Encode audio
latent, mu, var = model.encode(audio_waveform)

# Classify genre
_, class_out, _, _ = model(audio_waveform)
predicted_genre = class_out.argmax(dim=1)

# Generate/reconstruct audio
reconstructed_audio = model.generate(latent)
```

## Requirements

See `requirements.txt` for the complete list. Key dependencies include:
- PyTorch 2.9.0
- torchaudio 2.9.0
- torchcodec 0.8.1
- librosa, soundfile (audio processing)
- mlflow (experiment tracking)
- numpy, pandas (data handling)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

This project uses various open-source libraries for audio processing and deep learning. Special thanks to the PyTorch and MLflow communities.

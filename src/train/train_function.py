import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import os


def train_model(
    model,
    dataloader,
    num_epochs=20,
    device="cuda",
    lr=1e-4,
    beta_kl=0.001,
    recon_weight=1.0,
    class_weight=1.0,
    log_interval=10,
    log_dir="logs"
):
    """
    Train the EncoderDecoderModel on the given dataset.
    Creates:
        - batch_log.csv (per batch)
        - performance_log.csv (every 10 batches)
    """

    os.makedirs(log_dir, exist_ok=True)
    batch_log_path = os.path.join(log_dir, "batch_log.csv")
    perf_log_path = os.path.join(log_dir, "performance_log.csv")

    # Initialize CSV logs
    with open(batch_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "total_loss",
                        "recon_loss", "kl_loss", "class_loss", "class_acc"])
    with open(perf_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "recon_acc", "class_acc"])

    # Loss functions
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    for epoch in range(num_epochs):
        model.train()
        total_loss, recon_loss_total, kl_loss_total, class_loss_total = 0, 0, 0, 0
        total_acc = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (waveform, labels) in enumerate(progress_bar):
            waveform = waveform.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, class_out, mu, logvar = model(waveform)

            # --- Spectrogram Target ---
            with torch.no_grad():
                freq, _ = model.spectrogram(waveform)
                target = freq[:, :2, :, :]

            # --- Loss Components ---
            recon_loss = recon_criterion(recon, target)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = class_criterion(class_out, labels)

            # --- Total Loss ---
            loss = (
                recon_weight * recon_loss
                + beta_kl * kl_loss
                + class_weight * class_loss
            )

            loss.backward()
            optimizer.step()

            # --- Classification Accuracy ---
            preds = class_out.argmax(dim=1)
            class_acc = (preds == labels).float().mean().item()

            # --- Reconstruction Accuracy (cosine sim) ---
            with torch.no_grad():
                recon_flat = recon.flatten(start_dim=1)
                target_flat = target.flatten(start_dim=1)
                recon_acc = cos_sim(recon_flat, target_flat).mean().item()

            # --- Logging ---
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            class_loss_total += class_loss.item()
            total_acc += class_acc

            # Update batch log
            with open(batch_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    recon_loss.item(),
                    kl_loss.item(),
                    class_loss.item(),
                    class_acc
                ])

            # Performance log every 10 batches
            if batch_idx % log_interval == 0:
                with open(perf_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        batch_idx + 1,
                        recon_acc,
                        class_acc
                    ])
                progress_bar.set_postfix({
                    "Total": f"{total_loss / (batch_idx + 1):.4f}",
                    "Recon": f"{recon_loss_total / (batch_idx + 1):.4f}",
                    "KL": f"{kl_loss_total / (batch_idx + 1):.4f}",
                    "Class": f"{class_loss_total / (batch_idx + 1):.4f}",
                    "Acc": f"{total_acc / (batch_idx + 1):.4f}",
                })
        torch.mps.synchronize()
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Total Loss: {total_loss/len(dataloader):.4f}")
        print(f"  Recon Loss: {recon_loss_total/len(dataloader):.4f}")
        print(f"  KL Loss: {kl_loss_total/len(dataloader):.4f}")
        print(f"  Class Loss: {class_loss_total/len(dataloader):.4f}")
        print(f"  Class Acc: {total_acc/len(dataloader):.4f}")

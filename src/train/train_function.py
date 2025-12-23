import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # Import AMP
from tqdm import tqdm
import csv
import os
from typing import Optional

# Optional GCS upload support
try:
    from google.cloud import storage
except Exception:
    storage = None


def train_model(
    model,
    dataloader,
    val_dataloader=None,
    num_epochs=20,
    device="cuda",
    lr=1e-4,
    beta_kl=0.001,
    recon_weight=1.0,
    class_weight=1.0,
    log_interval=10,
    log_dir="logs",
    upload_best_to_gcs: bool = False,
    upload_all_epochs_to_gcs: bool = False,
    gcs_bucket: Optional[str] = None,
    gcs_dest_prefix: Optional[str] = None,
    create_archive: bool = False,
    archive_path: Optional[str] = None,
    accumulation_steps: int = 25  # Moved to arg for clarity
):
    os.makedirs(log_dir, exist_ok=True)
    batch_log_path = os.path.join(log_dir, "batch_log.csv")
    perf_log_path = os.path.join(log_dir, "performance_log.csv")

    # Initialize Logs
    for path in [batch_log_path, perf_log_path]:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Headers adapted slightly for clarity
            if "batch" in path:
                writer.writerow(
                    ["epoch", "batch", "loss", "recon_loss", "kl_loss"])
            else:
                writer.writerow(["epoch", "batch", "recon_acc", "class_acc"])

    # Setup Model & AMP
    # Keep model in FP32 initially, let AMP handle casting
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()  # For Mixed Precision

    recon_criterion = nn.MSELoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    best_acc = -1.0
    best_model_path = None

    # ---- Helper: Upload file to GCS ----
    def upload_to_gcs(local_path: str, dest_name: str):
        if storage and gcs_bucket:
            try:
                bucket = storage.Client().bucket(gcs_bucket)
                blob = bucket.blob(dest_name)
                blob.upload_from_filename(local_path)
                print(f"Uploaded to gs://{gcs_bucket}/{dest_name}")
            except Exception as e:
                print(f"GCS upload failed: {e}")

    print(f"Starting training on {device} with AMP enabled...")

    for epoch in range(num_epochs):
        model.train()

        # Open log file ONCE per epoch to reduce I/O overhead
        batch_log_file = open(batch_log_path, "a", newline="")
        batch_writer = csv.writer(batch_log_file)

        # Performance logging often happens less frequently, can keep separate or open/close
        perf_log_file = open(perf_log_path, "a", newline="")
        perf_writer = csv.writer(perf_log_file)

        # Trackers for average epoch metrics
        running_recon_acc = 0.0
        running_loss = 0.0

        optimizer.zero_grad()  # Initialize gradients once

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (waveform, labels) in enumerate(progress_bar):
            # Non-blocking transfer if pin_memory=True in dataloader
            waveform = waveform.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- Mixed Precision Context ---
            with autocast():
                recon, mu, logvar, target = model(waveform)
                target = target[:, 0:2, :, :]

                print(
                    f"Reconstructed shape: {recon.shape}, Target shape: {target.shape}")
                recon_loss = recon_criterion(recon, target)
                kl_loss = -0.5 * \
                    torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_weight * recon_loss + beta_kl * kl_loss

                # Normalize loss for gradient accumulation to keep magnitude consistent
                loss = loss / accumulation_steps

            # --- Backward & Optimizer Step ---
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # --- Logging Logic (Optimized) ---
            # We multiply loss back by accumulation_steps strictly for logging display
            current_loss_val = loss.item() * accumulation_steps

            # Buffer write to CSV (OS handles buffering, but avoiding open() is key)
            batch_writer.writerow([
                epoch + 1, batch_idx + 1,
                f"{current_loss_val:.4f}",
                f"{recon_loss.item():.4f}",
                f"{kl_loss.item():.4f}"
            ])

            # Calculate heavy metrics LESS frequently
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    # Compute cosine similarity
                    recon_flat = recon.flatten(start_dim=1)
                    target_flat = target.flatten(start_dim=1)
                    recon_acc = cos_sim(recon_flat, target_flat).mean().item()

                    running_recon_acc += recon_acc

                    perf_writer.writerow([
                        epoch + 1, batch_idx + 1, f"{recon_acc:.4f}", 0.0
                    ])

                    # Update progress bar occasionally, not every step
                    progress_bar.set_postfix(
                        {"Loss": f"{current_loss_val:.3f}", "RecAcc": f"{recon_acc:.3f}"})

        # Close file handles at end of epoch
        batch_log_file.close()
        perf_log_file.close()

        # Calculate epoch average for BEST model logic
        # Note: This is a rough average based on log_interval samples
        epoch_acc = running_recon_acc / (len(dataloader) / log_interval)

        print(f"Epoch {epoch+1} Complete. Approx Acc: {epoch_acc:.4f}")

        # --- Saving & Uploading (unchanged logic) ---
        epoch_model_path = os.path.join(log_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)

        if upload_all_epochs_to_gcs:
            dest_name = f"{gcs_dest_prefix.rstrip('/')}/epoch_{epoch+1}.pt" if gcs_dest_prefix else f"epoch_{epoch+1}.pt"
            upload_to_gcs(epoch_model_path, dest_name)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(log_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            if upload_best_to_gcs:
                dest_name = f"{gcs_dest_prefix.rstrip('/')}/best_model.pt" if gcs_dest_prefix else "best_model.pt"
                upload_to_gcs(best_model_path, dest_name)

    if create_archive:
        import tarfile
        from datetime import datetime
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        out_path = archive_path or f"logs_archive_{ts}.tar.gz"
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(log_dir, arcname=os.path.basename(log_dir))
        print(f"Created logs archive: {out_path}")

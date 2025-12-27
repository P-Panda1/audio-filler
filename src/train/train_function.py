import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # Import AMP
from tqdm import tqdm
import csv
import os
from typing import Optional
from src.blocks.SpectrogramBlock import SpectrogramBlock

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
    accumulation_steps: int = 1,  # Moved to arg for clarity
    val_split: float = 0.1,
    spectogram_model: Optional[SpectrogramBlock] = None,
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
    # scaler = GradScaler()  # For Mixed Precision

    cos_sim = nn.CosineSimilarity(dim=1)

    def complex_hybrid_loss(recon, target, alpha=0.7):
        # Convert [B,2,H,W] to complex [B,H,W]
        recon_c = torch.view_as_complex(recon.permute(0, 2, 3, 1))
        target_c = torch.view_as_complex(target.permute(0, 2, 3, 1))

        # Magnitude loss
        mag_loss = nn.L1Loss()(torch.abs(recon_c), torch.abs(target_c))

        # Phase-ish similarity (flattened)
        recon_flat = recon.flatten(1)
        target_flat = target.flatten(1)
        phase_loss = 1 - cos_sim(recon_flat, target_flat).mean()

        return alpha * mag_loss + (1 - alpha) * phase_loss

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

    if spectogram_model:
        spectogram_model = spectogram_model.to(device)

    print(f"Starting training on {device}")
    for batch_idx, (waveform, labels) in enumerate(dataloader):
        # Non-blocking transfer if pin_memory=True in dataloader
        waveform = waveform.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Determine split index
        if val_split > 0.0:
            val_size = int(waveform.size(0) * val_split)
            train_size = waveform.size(0) - val_size

            train_waveform = waveform[:train_size]
            train_labels = labels[:train_size]

            val_waveform = waveform[train_size:]
            val_labels = labels[train_size:]
        else:
            train_waveform = waveform
            train_labels = labels
            val_waveform = None
            val_labels = None

        # Apply spectrogram preprocessing if provided
        x_train = spectogram_model(
            train_waveform) if spectogram_model else train_waveform
        x_val = spectogram_model(val_waveform) if (
            spectogram_model and val_waveform is not None) else val_waveform

        model_mode = "train" if spectogram_model else "default"

        model.train()

        # Open log file ONCE per batch to reduce I/O overhead
        batch_log_file = open(batch_log_path, "a", newline="")
        batch_writer = csv.writer(batch_log_file)

        # Performance logging often happens less frequently, can keep separate or open/close
        perf_log_file = open(perf_log_path, "a", newline="")
        perf_writer = csv.writer(perf_log_file)

        # Trackers for average batch metrics
        running_recon_acc = 0.0
        running_loss = 0.0

        optimizer.zero_grad()  # Initialize gradients once

        progress_bar = tqdm(range(num_epochs),
                            desc=f"batch {batch_idx+1}/{len(dataloader)}")
        accum_counter = 0
        for epoch in progress_bar:

            # --- Standard Training Step (no AMP, no accumulation) ---
            recon, mu, logvar, target = model(x_train)
            target = target[:, 0:2, :, :]

            # For data parallelisation
            if target.dim() == 5:
                target = target.squeeze(2)
            if recon.dim() == 5:
                recon = recon.squeeze(2)

            recon_loss = complex_hybrid_loss(recon, target)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_weight * recon_loss + beta_kl * kl_loss

            # Backward & optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Logging Logic (Optimized) ---
            # We multiply loss back by accumulation_steps strictly for logging display
            current_loss_val = loss.item() * accumulation_steps

            recon_acc_val = "N/A"
            # Validation step
            if val_waveform is not None:
                model.eval()
                with torch.no_grad(), autocast():
                    recon_val, mu_val, logvar_val, target_val = model(
                        x_val)
                    target_val = target_val[:, 0:2, :, :]
                    if target_val.dim() == 5:
                        target_val = target_val.squeeze(2)
                    if recon_val.dim() == 5:
                        recon_val = recon_val.squeeze(2)
                    recon_acc_val = cos_sim(recon_val.flatten(start_dim=1),
                                            target_val.flatten(start_dim=1)).mean().item()

            # Buffer write to CSV (OS handles buffering, but avoiding open() is key)
            batch_writer.writerow([
                epoch + 1, batch_idx + 1,
                f"{current_loss_val:.4f}",
                f"{recon_loss.item():.4f}",
                f"{kl_loss.item():.4f}",
                f"{recon_acc_val:.4f}" if val_waveform is not None else "N/A"
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
                        {"Loss": f"{current_loss_val:.3f}", "RecAcc": f"{recon_acc:.3f}", "Epoch": epoch + 1, "ValAcc": f"{recon_acc_val:.3f}" if val_waveform is not None else "N/A"})

        # Close file handles at end of batch_idx
        batch_log_file.close()
        perf_log_file.close()

        # Calculate batch_idx average for BEST model logic
        # Note: This is a rough average based on log_interval samples
        batch_idx_acc = running_recon_acc / (len(dataloader) / log_interval)

        print(
            f"batch_idx {batch_idx+1} Complete. Approx Acc: {batch_idx_acc:.4f}")

        # # --- Saving & Uploading (unchanged logic) ---
        # batch_idx_model_path = os.path.join(
        #     log_dir, f"model_batch_idx{batch_idx+1}.pt")
        # torch.save(model.state_dict(), batch_idx_model_path)

        # if upload_all_epochs_to_gcs:
        #     dest_name = f"{gcs_dest_prefix.rstrip('/')}/batch_idx_{batch_idx+1}.pt" if gcs_dest_prefix else f"batch_idx_{batch_idx+1}.pt"
        #     upload_to_gcs(batch_idx_model_path, dest_name)

        if batch_idx_acc >= best_acc and batch_idx % 50 == 0:
            best_acc = batch_idx_acc
            best_model_path = os.path.join(log_dir, "best_model.pt")
            try:
                torch.save(model.state_dict(), best_model_path)
            except RuntimeError as e:
                print(f"⚠️ Skipping save at batch {batch_idx+1}: {e}")

            if upload_best_to_gcs:
                print(
                    f"Uploading best model with Acc: {best_acc:.4f} to GCS...")
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

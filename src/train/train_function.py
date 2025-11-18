import torch
import torch.nn as nn
import torch.optim as optim
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
    # Optional: upload best model to GCS after training
    upload_best_to_gcs: bool = False,
    gcs_bucket: Optional[str] = None,
    gcs_dest_prefix: Optional[str] = None,
    # Optional: create a tar.gz archive of the logs directory for easy scp/rsync
    create_archive: bool = False,
    archive_path: Optional[str] = None,
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

    import tarfile
    from datetime import datetime

    # Best model tracking
    best_acc = -1.0
    best_model_path = None

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
        # Device-specific synchronization (best-effort)
        try:
            if device and "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception:
            pass

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        n_batches = len(dataloader) if len(dataloader) > 0 else 1
        print(f"  Total Loss: {total_loss/n_batches:.4f}")
        print(f"  Recon Loss: {recon_loss_total/n_batches:.4f}")
        print(f"  KL Loss: {kl_loss_total/n_batches:.4f}")
        print(f"  Class Loss: {class_loss_total/n_batches:.4f}")
        avg_class_acc = total_acc / n_batches
        print(f"  Class Acc: {avg_class_acc:.4f}")

        # Run validation if provided
        if val_dataloader is not None:
            try:
                model.eval()
                val_total_acc = 0.0
                val_recon_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for v_waveform, v_labels in val_dataloader:
                        v_waveform = v_waveform.to(device)
                        v_labels = v_labels.to(device)
                        v_recon, v_class_out, v_mu, v_logvar = model(
                            v_waveform)
                        # spectrogram target
                        freq, _ = model.spectrogram(v_waveform)
                        target = freq[:, :2, :, :]
                        v_recon_loss = recon_criterion(v_recon, target)
                        preds = v_class_out.argmax(dim=1)
                        v_class_acc = (preds == v_labels).float().mean().item()
                        val_total_acc += v_class_acc
                        val_recon_total += v_recon_loss.item()
                        val_batches += 1
                if val_batches > 0:
                    val_acc = val_total_acc / val_batches
                    val_recon = val_recon_total / val_batches
                    print(
                        f"\nValidation: Recon Loss: {val_recon:.4f} | Class Acc: {val_acc:.4f}")
            except Exception as e:
                print(f"Validation run failed: {e}")

        # Save best model by class accuracy
        if avg_class_acc > best_acc:
            best_acc = avg_class_acc
            best_model_path = os.path.join(
                log_dir, f"best_model_epoch{epoch+1}.pt")
            try:
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"Saved best model to {best_model_path} (class_acc={best_acc:.4f})")
            except Exception as e:
                print(f"Failed to save best model: {e}")

    # After training: optionally upload best model to GCS
    if upload_best_to_gcs and best_model_path is not None:
        if storage is None:
            print("google-cloud-storage is not installed; cannot upload to GCS.")
        elif not gcs_bucket:
            print("gcs_bucket not provided; skipping upload to GCS.")
        else:
            try:
                client = storage.Client()
                bucket = client.bucket(gcs_bucket)
                dest_prefix = gcs_dest_prefix.rstrip(
                    '/') if gcs_dest_prefix else ''
                dest_blob = f"{dest_prefix}/{os.path.basename(best_model_path)}" if dest_prefix else os.path.basename(
                    best_model_path)
                blob = bucket.blob(dest_blob)
                blob.upload_from_filename(best_model_path)
                print(f"Uploaded best model to gs://{gcs_bucket}/{dest_blob}")
            except Exception as e:
                print(f"Failed to upload best model to GCS: {e}")

    # Optionally create an archive of the logs directory for easy download
    if create_archive:
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            default_archive = os.path.join(os.path.dirname(
                log_dir), f"logs_archive_{timestamp}.tar.gz")
            out_path = archive_path if archive_path else default_archive
            with tarfile.open(out_path, "w:gz") as tar:
                # add the entire log_dir
                tar.add(log_dir, arcname=os.path.basename(log_dir))
            print(f"Created archive of logs at: {out_path}")
        except Exception as e:
            print(f"Failed to create archive of logs: {e}")

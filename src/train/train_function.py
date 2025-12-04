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
    upload_best_to_gcs: bool = False,
    upload_all_epochs_to_gcs: bool = False,   # <-- NEW
    gcs_bucket: Optional[str] = None,
    gcs_dest_prefix: Optional[str] = None,
    create_archive: bool = False,
    archive_path: Optional[str] = None,
):
    """
    Train the EncoderDecoderModel.
    Now supports uploading:
        - best model only
        - OR every epoch model
    """

    os.makedirs(log_dir, exist_ok=True)
    batch_log_path = os.path.join(log_dir, "batch_log.csv")
    perf_log_path = os.path.join(log_dir, "performance_log.csv")

    # Init CSV files
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
    class_criterion = nn.CrossEntropyLoss()   # (still unused)
    cos_sim = nn.CosineSimilarity(dim=1)

    best_acc = -1.0
    best_model_path = None

    # ---- Helper: Upload file to GCS ----
    def upload_to_gcs(local_path: str, dest_name: str):
        if storage is None:
            print("GCS upload requested but google-cloud-storage not installed.")
            return
        if not gcs_bucket:
            print("GCS bucket not provided; skipping upload.")
            return
        try:
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)
            blob = bucket.blob(dest_name)
            blob.upload_from_filename(local_path)
            print(f"Uploaded to gs://{gcs_bucket}/{dest_name}")
        except Exception as e:
            print(f"GCS upload failed: {e}")

    # =============================================
    # TRAINING LOOP
    # =============================================
    for epoch in range(num_epochs):
        model.train()
        total_loss = recon_loss_total = kl_loss_total = 0.0
        total_acc = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (waveform, labels) in enumerate(progress_bar):
            waveform = waveform.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            recon, mu, logvar = model(waveform)

            # Spectrogram target
            with torch.no_grad():
                freq, _ = model.spectrogram(waveform)
                target = freq[:, :2, :, :]

            # Losses
            recon_loss = recon_criterion(recon, target)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_weight * recon_loss + beta_kl * kl_loss

            loss.backward()
            optimizer.step()

            # Fake class_acc (your class head is not active)
            class_acc = 0.0

            with torch.no_grad():
                recon_flat = recon.flatten(start_dim=1)
                target_flat = target.flatten(start_dim=1)
                recon_acc = cos_sim(recon_flat, target_flat).mean().item()

            # logs
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            total_acc += class_acc

            with open(batch_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, batch_idx + 1,
                    loss.item(), recon_loss.item(), kl_loss.item(),
                    class_acc
                ])

            # perf log
            if batch_idx % log_interval == 0:
                with open(perf_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, batch_idx + 1, recon_acc, class_acc
                    ])

        # END OF EPOCH
        n_batches = len(dataloader)
        epoch_acc = total_acc / max(1, n_batches)

        print(f"Epoch {epoch+1}: avg_acc={epoch_acc:.4f}")

        # ------------------------------------------------
        # (A) SAVE LOCAL WEIGHTS FOR THIS EPOCH
        # ------------------------------------------------
        epoch_model_path = os.path.join(log_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Saved epoch model to {epoch_model_path}")

        # ------------------------------------------------
        # (B) UPLOAD THIS EPOCH TO GCS (optional)
        # ------------------------------------------------
        if upload_all_epochs_to_gcs:
            dest_name = (
                f"{gcs_dest_prefix.rstrip('/')}/epoch_{epoch+1}.pt"
                if gcs_dest_prefix else f"epoch_{epoch+1}.pt"
            )
            upload_to_gcs(epoch_model_path, dest_name)

        # ------------------------------------------------
        # (C) TRACK BEST MODEL
        # ------------------------------------------------
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_path = os.path.join(log_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print("Updated BEST model.")

    # END TRAINING LOOP

    # ------------------------------------------------
    # (D) UPLOAD BEST MODEL
    # ------------------------------------------------
    if upload_best_to_gcs and best_model_path is not None:
        dest_name = (
            f"{gcs_dest_prefix.rstrip('/')}/best_model.pt"
            if gcs_dest_prefix else "best_model.pt"
        )
        upload_to_gcs(best_model_path, dest_name)

    # ------------------------------------------------
    # (E) Create archive if needed
    # ------------------------------------------------
    if create_archive:
        import tarfile
        from datetime import datetime
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        out_path = archive_path or f"logs_archive_{ts}.tar.gz"
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(log_dir, arcname=os.path.basename(log_dir))
        print(f"Created logs archive: {out_path}")

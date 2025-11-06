import mlflow
import mlflow.pytorch
from src.train.train_function import train_model


def run_experiment(
    model,
    dataloader,
    num_epochs=20,
    lr=1e-4,
    beta_kl=0.001,
    recon_weight=1.0,
    class_weight=1.0,
    device="cuda",
    experiment_name="MusicVAE",
    log_dir="logs",
):
    """
    Wrapper to launch and track a training run in MLflow.
    Calls the training loop defined in trainer.py.
    """

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"lr{lr}_rw{recon_weight}_cw{class_weight}_kl{beta_kl}"):
        # --- Log hyperparameters ---
        mlflow.log_params({
            "num_epochs": num_epochs,
            "lr": lr,
            "beta_kl": beta_kl,
            "recon_weight": recon_weight,
            "class_weight": class_weight,
            "device": device,
        })

        # --- Run training loop ---
        train_model(
            model=model,
            dataloader=dataloader,
            num_epochs=num_epochs,
            device=device,
            lr=lr,
            beta_kl=beta_kl,
            recon_weight=recon_weight,
            class_weight=class_weight,
            log_interval=10,
            log_dir=log_dir
        )

        # --- Log logs and model artifacts ---
        mlflow.log_artifact(f"{log_dir}/batch_log.csv")
        mlflow.log_artifact(f"{log_dir}/performance_log.csv")

        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"✅ Run completed. Check MLflow UI for results.")

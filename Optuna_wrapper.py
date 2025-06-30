import subprocess, json, pathlib, optuna, uuid, tempfile

SCRIPT = "Dual_parametric.py"

def objective(trial):
    lr  = trial.suggest_loguniform("hparams.lr", 1e-5, 1e-2)
    bs  = trial.suggest_categorical("hparams.batch_size_train", [512, 1024, 2048])

    metric_path = pathlib.Path(tempfile.gettempdir()) / f"metric_{uuid.uuid4()}.json"

    cmd = [
        "python", SCRIPT,
        f"hparams.lr={lr}",
        f"hparams.batch_size_train={bs}",
        f"hparams.metric_path={metric_path}"
    ]
    subprocess.run(cmd, check=True)

    with metric_path.open() as f:
        return json.load(f)["val_error"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, show_progress_bar=True)
print("best metric:", study.best_value)
print("best params:", study.best_params)

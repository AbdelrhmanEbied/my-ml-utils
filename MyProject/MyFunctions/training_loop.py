import copy
import warnings
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple


def training_loop(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  epochs: int = 1,
                  scheduler=None,                        
                  step_scheduler_per_batch: bool = False,
                  use_amp: bool = False,
                  max_grad_norm: float = None,
                  checkpoint_dir: str = "checkpoints",
                  checkpoint_name: str = "best_model.pth",
                  patience: int = 5,
                  min_delta: float = 0.0,
                  restore_best_weights: bool = True,
                  metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                  metric_name: str = "acc") -> Tuple[Dict[str, list], plt.Figure]:
    """
    A general-purpose PyTorch training loop with:
      - Mixed-precision (AMP) support
      - Gradient clipping
      - Per-batch or per-epoch LR scheduling (including ReduceLROnPlateau)
      - Checkpoint saving (best model by test loss)
      - Early stopping with optional best-weight restoration
      - Optional custom metric tracking
      - Loss + metric curve plotting

    Args:
        model:                  Model to train.
        train_dataloader:       DataLoader for training data.
        test_dataloader:        DataLoader for evaluation data.
        loss_fn:                Loss function. Must accept (y_pred, y_true).
        optimizer:              Optimizer instance.
        device:                 Target device (cpu / cuda / mps).
        epochs:                 Maximum number of training epochs.
        scheduler:              Optional LR scheduler. Pass None to disable.
        step_scheduler_per_batch: If True, step() is called after each batch
                                  instead of each epoch. Has no effect on
                                  ReduceLROnPlateau (always stepped per epoch).
        use_amp:                Enable automatic mixed precision. Only active
                                on CUDA; a warning is raised on other devices.
        max_grad_norm:          If set, clips gradient norm to this value.
        checkpoint_dir:         Directory where the best checkpoint is saved.
        checkpoint_name:        Filename for the best checkpoint.
        patience:               Early stopping patience (epochs without improvement).
        min_delta:              Minimum improvement in test loss to reset patience.
        restore_best_weights:   If True, load the best checkpoint into the model
                                before returning (recommended when early stopping
                                is used).
        metric_fn:              Optional callable (y_pred_raw, y_true) -> float.
                                Receives *raw model outputs* (logits / probabilities)
                                — apply argmax / threshold inside the function as
                                needed.
        metric_name:            Display name for the custom metric (e.g. "acc").

    Returns:
        results:  Dict with keys train_loss, test_loss, and optionally
                  train_{metric_name} / test_{metric_name}.
        fig:      Matplotlib Figure with loss (and metric) curves. The figure
                  is closed before returning so it does not auto-display;
                  call fig.show() or fig.savefig() as needed.
    """

    #  setup
    if use_amp and device.type != "cuda":
        warnings.warn(
            f"use_amp=True has no effect on device '{device.type}'. "
            "AMP is only supported on CUDA. Falling back to full precision.",
            UserWarning,
            stacklevel=2,
        )
        use_amp = False

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_path / checkpoint_name

    results: Dict[str, list] = {"train_loss": [], "test_loss": []}
    if metric_fn:
        results[f"train_{metric_name}"] = []
        results[f"test_{metric_name}"] = []

    best_test_loss = float("inf")
    epochs_no_improve = 0

    # GradScaler — keyword arg required; only instantiated for CUDA + AMP
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    main_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in main_bar:

        # train phase
        model.train()
        train_loss, train_metric_sum, train_samples = 0.0, 0.0, 0

        for X, y in train_dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

            if scaler:
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()


            if (scheduler
                    and step_scheduler_per_batch
                    and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step()

            batch_size = len(y)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

            if metric_fn:
                with torch.no_grad():   
                    train_metric_sum += metric_fn(y_pred.detach(), y) * batch_size

        epoch_train_loss = train_loss / train_samples
        results["train_loss"].append(epoch_train_loss)
        if metric_fn:
            results[f"train_{metric_name}"].append(train_metric_sum / train_samples)

        # eval phase
        model.eval()
        test_loss, test_metric_sum, test_samples = 0.0, 0.0, 0

        with torch.inference_mode():
            for X, y in test_dataloader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)

                batch_size = len(y)
                test_loss += loss.item() * batch_size
                test_samples += batch_size

                if metric_fn:
                    test_metric_sum += metric_fn(y_pred, y) * batch_size

        epoch_test_loss = test_loss / test_samples
        results["test_loss"].append(epoch_test_loss)
        if metric_fn:
            results[f"test_{metric_name}"].append(test_metric_sum / test_samples)

        # scheduler (per-epoch path)
        if scheduler and not step_scheduler_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_test_loss)
            else:
                scheduler.step()

        # checkpoint + early stopping
        if epoch_test_loss < (best_test_loss - min_delta):
            best_test_loss = epoch_test_loss
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "loss": best_test_loss,

                "results": copy.deepcopy(results),
            }
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1

         # progress bar
        postfix = {
            "Tr_Loss": f"{epoch_train_loss:.4f}",
            "Te_Loss": f"{epoch_test_loss:.4f}",
            "Patience": f"{epochs_no_improve}/{patience}",
        }
        if metric_fn:
            postfix[f"Te_{metric_name.capitalize()}"] = (
                f"{test_metric_sum / test_samples:.4f}"
            )
        main_bar.set_postfix(postfix)

        if epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}.")
            break


    if restore_best_weights and save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] Restored best weights from epoch {checkpoint['epoch'] + 1} "
              f"(test loss: {checkpoint['loss']:.4f}).")

    # plotting
    actual_epochs = len(results["train_loss"])
    epochs_range = range(1, actual_epochs + 1)

    num_plots = 2 if metric_fn else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]   

    axes[0].plot(epochs_range, results["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs_range, results["test_loss"],  label="Test",  marker="o")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if metric_fn:
        axes[1].plot(epochs_range, results[f"train_{metric_name}"], label="Train", marker="o")
        axes[1].plot(epochs_range, results[f"test_{metric_name}"],  label="Test",  marker="o")
        axes[1].set_title(f"{metric_name.capitalize()} curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_name.capitalize())
        axes[1].legend()

    plt.tight_layout()
    plt.close(fig)   

    return results, fig

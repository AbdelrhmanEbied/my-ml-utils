import copy
import warnings
import torch
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Tuple


def training_loop(
    model: torch.nn.Module,
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
    metric_name: str = "acc",
    accumulation_steps: int = 1,
    model_ema: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> Dict[str, list]:
    """
    A production-grade PyTorch training loop designed for large models such as
    Vision Transformers (ViTs). Extends the base loop with gradient accumulation,
    Exponential Moving Average (EMA) weight tracking, and headless experiment
    logging.

    Base features (logic unchanged from original):
      - Mixed-precision (AMP) support via autocast / GradScaler
      - Gradient clipping
      - Per-batch or per-epoch LR scheduling (including ReduceLROnPlateau)
      - Checkpoint saving (best model by test loss)
      - Early stopping with optional best-weight restoration
      - Optional custom metric tracking

    New features:
      - Gradient accumulation across N micro-batches before a weight update
      - EMA shadow model updated after every optimizer step
      - Headless experiment logging via any object exposing a ``.log()`` method
        (e.g. ``wandb``, ``mlflow``, a custom logger)

    Args:
        model (torch.nn.Module):
            Model to train.
        train_dataloader (torch.utils.data.DataLoader):
            DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader):
            DataLoader for evaluation data.
        loss_fn (torch.nn.Module):
            Loss function. Must accept ``(y_pred, y_true)``.
        optimizer (torch.optim.Optimizer):
            Optimizer instance.
        device (torch.device):
            Target device (cpu / cuda / mps).
        epochs (int):
            Maximum number of training epochs. Default: 1.
        scheduler:
            Optional LR scheduler. Pass ``None`` to disable.
        step_scheduler_per_batch (bool):
            If ``True``, ``scheduler.step()`` is called after every optimizer
            step instead of each epoch. Has no effect on
            ``ReduceLROnPlateau`` (always stepped per epoch). Default: False.
        use_amp (bool):
            Enable automatic mixed precision. Only active on CUDA; a warning
            is raised on other devices. Default: False.
        max_grad_norm (float | None):
            If set, clips gradient norm to this value before each optimizer
            step. Default: None.
        checkpoint_dir (str):
            Directory where the best checkpoint is saved. Default: "checkpoints".
        checkpoint_name (str):
            Filename for the best checkpoint. Default: "best_model.pth".
        patience (int):
            Early stopping patience - number of epochs without improvement
            before training is halted. Default: 5.
        min_delta (float):
            Minimum improvement in test loss required to reset patience.
            Default: 0.0.
        restore_best_weights (bool):
            If ``True``, load the best checkpoint into the model before
            returning. Recommended when early stopping is enabled.
            Default: True.
        metric_fn (Callable[[torch.Tensor, torch.Tensor], float] | None):
            Optional callable ``(y_pred_raw, y_true) -> float``.
            Receives *raw model outputs* (logits / probabilities) - apply
            ``argmax`` / threshold inside the function as needed.
            Default: None.
        metric_name (str):
            Display name for the custom metric (e.g. ``"acc"``). Default: "acc".
        accumulation_steps (int):
            Number of micro-batches to accumulate gradients over before
            performing a single optimizer step. The loss is normalized by
            this factor so the effective gradient magnitude is independent of
            the accumulation window. Must be >= 1. Default: 1 (no accumulation).
        model_ema (timm.utils.ModelEmaV2 | None):
            Optional Exponential Moving Average shadow model. Expected interface:
            ``model_ema.update(model)`` called after every optimizer step,
            ``model_ema.module`` is the EMA-averaged nn.Module used for eval.
            If ``None``, EMA is disabled and the raw model is used for
            evaluation. Default: None.
        logger (Any | None):
            Optional experiment tracker with a ``.log(dict)`` method
            (e.g. ``wandb``, ``mlflow.ActiveRun``, a custom object).
            At the end of every epoch a dictionary is passed to ``logger.log()``
            containing: epoch (1-indexed), train_loss, test_loss, lr, and
            optionally train_{metric_name} / test_{metric_name}.
            Default: None.

    Returns:
        results (Dict[str, list]):
            Dictionary with keys ``train_loss``, ``test_loss``, and
            optionally ``train_{metric_name}`` / ``test_{metric_name}``.
    """

    # Setup
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}.")

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

    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    main_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in main_bar:

        # Train Phase
        model.train()
        train_loss, train_metric_sum, train_samples = 0.0, 0.0, 0
        num_batches = len(train_dataloader)

        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                y_pred = model(X)
                loss = loss_fn(y_pred, y) / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accumulation_boundary = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches

            if is_accumulation_boundary or is_last_batch:
                if scaler:
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                if model_ema is not None:
                    model_ema.update(model)

                optimizer.zero_grad()

                if (
                    scheduler
                    and step_scheduler_per_batch
                    and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                ):
                    scheduler.step()

            batch_size = len(y)
            train_loss += loss.item() * accumulation_steps * batch_size
            train_samples += batch_size

            if metric_fn:
                with torch.no_grad():
                    train_metric_sum += metric_fn(y_pred.detach().float(), y) * batch_size

        epoch_train_loss = train_loss / train_samples
        results["train_loss"].append(epoch_train_loss)
        if metric_fn:
            results[f"train_{metric_name}"].append(train_metric_sum / train_samples)


        # Eval Phase
        eval_model = model_ema.module if model_ema is not None else model
        eval_model.eval()

        test_loss, test_metric_sum, test_samples = 0.0, 0.0, 0

        with torch.inference_mode():
            for X, y in test_dataloader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    y_pred = eval_model(X)
                    loss = loss_fn(y_pred, y)

                batch_size = len(y)
                test_loss += loss.item() * batch_size
                test_samples += batch_size

                if metric_fn:
                    test_metric_sum += metric_fn(y_pred.float(), y) * batch_size

        epoch_test_loss = test_loss / test_samples
        results["test_loss"].append(epoch_test_loss)
        if metric_fn:
            results[f"test_{metric_name}"].append(test_metric_sum / test_samples)


        if scheduler and not step_scheduler_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_test_loss)
            else:
                scheduler.step()


        current_lr = optimizer.param_groups[0]["lr"]

        if logger is not None:
            log_dict: Dict[str, Any] = {
                "epoch":      epoch + 1,
                "train_loss": epoch_train_loss,
                "test_loss":  epoch_test_loss,
                "lr":         current_lr,
            }
            if metric_fn:
                log_dict[f"train_{metric_name}"] = results[f"train_{metric_name}"][-1]
                log_dict[f"test_{metric_name}"]  = results[f"test_{metric_name}"][-1]
            logger.log(log_dict)


        # Checkpointing
        if epoch_test_loss < (best_test_loss - min_delta):
            best_test_loss = epoch_test_loss
            epochs_no_improve = 0
            checkpoint = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "loss":                 best_test_loss,
                "results":              copy.deepcopy(results),
            }
            if model_ema is not None:
                checkpoint["model_ema_state_dict"] = model_ema.module.state_dict()
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1


        postfix = {
            "Tr_Loss":  f"{epoch_train_loss:.4f}",
            "Te_Loss":  f"{epoch_test_loss:.4f}",
            "LR":       f"{current_lr:.2e}",
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


    # Weight Restoration
    if restore_best_weights and save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"[INFO] Restored best weights from epoch {checkpoint['epoch'] + 1} "
            f"(test loss: {checkpoint['loss']:.4f})."
        )
        if model_ema is not None and "model_ema_state_dict" in checkpoint:
            model_ema.module.load_state_dict(checkpoint["model_ema_state_dict"])
            print("[INFO] Restored EMA weights from the best checkpoint.")

    return results

import os
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Protocol, runtime_checkable


# ── Protocols ──────────────────────────────────────────────────────────────────

@runtime_checkable
class _Logger(Protocol):
    def log(self, metrics: Dict[str, Any], step: int) -> None: ...

@runtime_checkable
class _ModelEMA(Protocol):
    module: torch.nn.Module
    def update(self, model: torch.nn.Module) -> None: ...


# ── CUDA Prefetcher ────────────────────────────────────────────────────────────

class _CudaPrefetcher:
    """
    Overlaps CPU→GPU transfer of batch N+1 with the forward/backward of batch N
    via a dedicated CUDA stream. Falls back to a plain iterator on non-CUDA devices.
    """
    def __init__(self, dataloader, device: torch.device):
        self._loader = dataloader
        self._device = device
        self._use    = device.type == "cuda"

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        if not self._use:
            for X, y in self._loader:
                yield X.to(self._device, non_blocking=True), y.to(self._device, non_blocking=True)
            return

        stream = torch.cuda.Stream()
        loader = iter(self._loader)
        X = y  = None

        def _prefetch():
            nonlocal X, y
            try:
                raw_X, raw_y = next(loader)
                with torch.cuda.stream(stream):
                    X = raw_X.to(self._device, non_blocking=True)
                    y = raw_y.to(self._device, non_blocking=True)
            except StopIteration:
                X = y = None

        _prefetch()
        while X is not None:
            torch.cuda.current_stream().wait_stream(stream)
            cur_X, cur_y = X, y
            _prefetch()
            yield cur_X, cur_y


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def _is_main() -> bool:
    return not _is_ddp() or dist.get_rank() == 0

def _reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if _is_ddp():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _safe_metric(
    metric_fn: Callable,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
   
    pred_cpu = y_pred.detach().float().cpu()
    true_cpu = y_true.detach().cpu()

    try:
        pred_np = pred_cpu.numpy()
        true_np = true_cpu.numpy()
        value   = float(metric_fn(pred_np, true_np))
    except (TypeError, AttributeError):
        value = float(metric_fn(pred_cpu, true_cpu))

    return torch.tensor(value, device=device, dtype=torch.float32)


def _train_one_epoch(
    model, dataloader, loss_fn, optimizer, device,
    scaler, max_grad_norm, accumulation_steps,
    model_ema, scheduler, step_scheduler_per_batch, use_amp, metric_fn,
) -> tuple:
    model.train()
    total_loss, metric_sum, total_samples = 0.0, 0.0, 0
    num_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (X, y) in enumerate(_CudaPrefetcher(dataloader, device)):
        is_accumulation_boundary = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch            = (batch_idx + 1) == num_batches
        is_partial_last_window   = is_last_batch and not is_accumulation_boundary

    
        remainder   = num_batches % accumulation_steps
        window_size = remainder if is_partial_last_window and remainder != 0 else accumulation_steps

        with torch.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(X)
            loss   = loss_fn(y_pred, y) / window_size

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_accumulation_boundary or is_last_batch:
            if scaler:
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_was_skipped = scaler.get_scale() < scale_before
            else:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                step_was_skipped = False

            if model_ema is not None and not step_was_skipped:
                model_ema.update(_unwrap(model))
            optimizer.zero_grad(set_to_none=True)

            if (
                scheduler
                and step_scheduler_per_batch
                and not step_was_skipped
                and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            ):
                scheduler.step()

        batch_size  = len(y)
        loss_scalar = _reduce_mean(loss.detach() * window_size).item()
        total_loss   += loss_scalar * batch_size
        total_samples += batch_size

        if metric_fn:
            with torch.inference_mode():
                m = _safe_metric(metric_fn, y_pred, y, device)
                metric_sum += _reduce_mean(m).item() * batch_size

    return total_loss / total_samples, (metric_sum / total_samples if metric_fn else None)


def _eval_one_epoch(
    model, dataloader, loss_fn, device, use_amp, metric_fn,
) -> tuple:
    model.eval()
    total_loss, metric_sum, total_samples = 0.0, 0.0, 0

    with torch.inference_mode():
        for X, y in _CudaPrefetcher(dataloader, device):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                y_pred = model(X)
                loss   = loss_fn(y_pred, y)

            batch_size  = len(y)
            loss_scalar = _reduce_mean(loss.detach()).item()
            total_loss   += loss_scalar * batch_size
            total_samples += batch_size

            if metric_fn:
                m = _safe_metric(metric_fn, y_pred, y, device)
                metric_sum += _reduce_mean(m).item() * batch_size

    return total_loss / total_samples, (metric_sum / total_samples if metric_fn else None)


# ── Trainer ────────────────────────────────────────────────────────────────────

def trainer(
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
    max_grad_norm: Optional[float] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "best_model.pth",
    patience: int = 5,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    metric_fn: Optional[Callable] = None,
    metric_name: str = "acc",
    accumulation_steps: int = 1,
    model_ema: Optional[_ModelEMA] = None,
    logger: Optional[_Logger] = None,
    monitor: str = "test_loss",
    monitor_mode: str = "min",
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    channels_last: bool = False,
    ddp_find_unused: bool = False,
    sampler=None,
) -> Dict[str, Any]:
    """
    Production-grade PyTorch training loop — v4.

    Critical fixes over v3:
      FIX 1  DDP early-stopping deadlock   — stop_flag tensor broadcast to all ranks.
      FIX 2  AMP-scheduler desync          — scheduler.step() gated on GradScaler scale.
      FIX 3  Cross-device metric crash     — metric_fn receives CPU/numpy, result re-synced.
      FIX 4  Gradient accumulation scaling — remainder window uses its actual size, not accumulation_steps.

    Retained optimizations:
      torch.compile, _CudaPrefetcher, DistributedDataParallel, channels-last,
      TF32 + cuDNN benchmark, zero_grad(set_to_none=True), all_reduce AVG sync.

    Args:
        model:                    Model to train. Compiled/wrapped internally.
        train_dataloader:         Training DataLoader (pin_memory=True recommended).
        test_dataloader:          Evaluation DataLoader.
        loss_fn:                  Loss callable (y_pred, y_true).
        optimizer:                Optimizer instance.
        device:                   Target device (cpu / cuda / cuda:N / mps).
        epochs:                   Maximum training epochs.
        scheduler:                Optional LR scheduler.
        step_scheduler_per_batch: Call scheduler.step() after each successful optimizer step.
        use_amp:                  Mixed precision — CUDA only.
        max_grad_norm:            Gradient clipping threshold. None disables it.
        checkpoint_dir:           Checkpoint directory.
        checkpoint_name:          Checkpoint filename.
        patience:                 Early-stopping patience in epochs.
        min_delta:                Minimum change to reset patience counter.
        restore_best_weights:     Reload best checkpoint before returning.
        metric_fn:                Callable (y_pred, y_true) -> float. Accepts numpy or tensors.
        metric_name:              Key and display name for the custom metric.
        accumulation_steps:       Micro-batches per optimizer step (>= 1).
        model_ema:                EMA shadow model (timm ModelEmaV2 interface).
        logger:                   Object with .log(dict, step=int).
        monitor:                  Metric key to monitor for checkpointing / early stop.
        monitor_mode:             "min" or "max".
        compile_model:            Wrap model with torch.compile before training.
        compile_mode:             "default" | "reduce-overhead" | "max-autotune".
        channels_last:            Convert model to NHWC layout (best for CNNs).
        ddp_find_unused:          DDP find_unused_parameters flag.
        sampler:                  DistributedSampler — set_epoch() called each epoch.

    Returns:
        results dict on rank 0; empty dict on non-main DDP ranks.
    """

    # ── Validation ─────────────────────────────────────────────────────────────
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}.")
    if monitor_mode not in ("min", "max"):
        raise ValueError(f"monitor_mode must be 'min' or 'max', got '{monitor_mode}'.")

    n_train = len(train_dataloader)
    if accumulation_steps > n_train:
        warnings.warn(
            f"accumulation_steps={accumulation_steps} > batches ({n_train}). "
            "Effective accumulation equals dataset size.",
            UserWarning, stacklevel=2,
        )

    if use_amp and device.type != "cuda":
        warnings.warn(
            f"use_amp=True has no effect on '{device.type}'. Falling back to fp32.",
            UserWarning, stacklevel=2,
        )
        use_amp = False

    # ── Hardware flags ─────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if compile_model:
        model = torch.compile(model, mode=compile_mode, fullgraph=False)

    if _is_ddp():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=ddp_find_unused)


    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)

    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    is_main   = _is_main()
    ckpt_path = Path(checkpoint_dir)
    if is_main:
        ckpt_path.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_path / checkpoint_name

    results: Dict[str, Any] = {"train_loss": [], "test_loss": [], "best_epoch": 0}
    if metric_fn:
        results[f"train_{metric_name}"] = []
        results[f"test_{metric_name}"]  = []

    is_better         = (lambda a, b: a < b - min_delta) if monitor_mode == "min" else (lambda a, b: a > b + min_delta)
    best_value        = float("inf") if monitor_mode == "min" else float("-inf")
    epochs_no_improve = 0
    main_bar          = tqdm(range(epochs), desc="Training Progress", disable=not is_main)

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in main_bar:
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_train_loss, train_metric = _train_one_epoch(
            model, train_dataloader, loss_fn, optimizer, device,
            scaler, max_grad_norm, accumulation_steps,
            model_ema, scheduler, step_scheduler_per_batch, use_amp, metric_fn,
        )

        eval_model = model_ema.module if model_ema is not None else _unwrap(model)
        epoch_test_loss, test_metric = _eval_one_epoch(
            eval_model, test_dataloader, loss_fn, device, use_amp, metric_fn,
        )

        # Non-main ranks skip all logging/checkpointing but MUST NOT skip the
        # stop_flag all_reduce below — they must stay in lock-step with rank 0.
        if is_main:
            results["train_loss"].append(epoch_train_loss)
            results["test_loss"].append(epoch_test_loss)
            if metric_fn:
                results[f"train_{metric_name}"].append(train_metric)
                results[f"test_{metric_name}"].append(test_metric)

            if scheduler and not step_scheduler_per_batch:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_test_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            if logger is not None:
                log_dict: Dict[str, Any] = {
                    "train_loss": epoch_train_loss,
                    "test_loss":  epoch_test_loss,
                    "lr":         current_lr,
                }
                if metric_fn:
                    log_dict[f"train_{metric_name}"] = train_metric
                    log_dict[f"test_{metric_name}"]  = test_metric
                logger.log(log_dict, step=epoch + 1)

            monitored_values: Dict[str, float] = {
                "train_loss": epoch_train_loss,
                "test_loss":  epoch_test_loss,
            }
            if metric_fn:
                monitored_values[f"train_{metric_name}"] = train_metric
                monitored_values[f"test_{metric_name}"]  = test_metric

            if monitor not in monitored_values:
                raise ValueError(f"monitor='{monitor}' not in {list(monitored_values)}.")

            current_value = monitored_values[monitor]

            if is_better(current_value, best_value):
                best_value            = current_value
                results["best_epoch"] = epoch + 1
                epochs_no_improve     = 0
                raw         = _unwrap(model)
                saved_state = model_ema.module.state_dict() if model_ema is not None else raw.state_dict()
                checkpoint  = {
                    "epoch":                epoch,
                    "model_state_dict":     saved_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    f"best_{monitor}":      best_value,
                    "results_snapshot":     {k: list(v) if isinstance(v, list) else v for k, v in results.items()},
                }
                try:
                    torch.save(checkpoint, save_path)
                except Exception as exc:
                    warnings.warn(f"Checkpoint save failed: {exc}", RuntimeWarning, stacklevel=2)
            else:
                epochs_no_improve += 1

            postfix = {
                "Tr_Loss":  f"{epoch_train_loss:.4f}",
                "Te_Loss":  f"{epoch_test_loss:.4f}",
                "LR":       f"{current_lr:.2e}",
                "Patience": f"{epochs_no_improve}/{patience}",
            }
            if metric_fn:
                postfix[f"Te_{metric_name.capitalize()}"] = f"{test_metric:.4f}"
            main_bar.set_postfix(postfix)

            if epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}.")
                stop_flag.fill_(1)

   
        if _is_ddp():
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

        if stop_flag.item() == 1:
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    if _is_ddp():
        dist.barrier()

    if restore_best_weights and is_main:
        if save_path.exists():
            checkpoint = torch.load(save_path, map_location=device, weights_only=True)
            _unwrap(model).load_state_dict(checkpoint["model_state_dict"])
            if model_ema is not None:
                model_ema.module.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"[INFO] Restored best weights from epoch {checkpoint['epoch'] + 1} "
                f"({monitor}: {checkpoint[f'best_{monitor}']:.4f})."
            )
        else:
            warnings.warn(
                "restore_best_weights=True but no checkpoint was saved. "
                "Returning weights from the final epoch.",
                UserWarning, stacklevel=2,
            )

    return results if is_main else {}

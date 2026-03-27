"""
load_checkpoint.py
==================
Utility for loading checkpoints saved by training_loop().
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
    weights_only: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint saved by training_loop() and restore state into
    the provided objects.

    Args:
        checkpoint_path:  Path to the .pth checkpoint file.
        model:            Model instance to restore weights into.
                          Must have the same architecture as when saved.
        optimizer:        Optional optimizer to restore state into.
                          Pass None to skip (e.g. inference-only use).
        scheduler:        Optional LR scheduler to restore state into.
                          Pass None to skip.
        device:           Device to map the checkpoint onto.
                          Defaults to the model's current device.
        weights_only:     If True, only model weights are loaded — optimizer
                          and scheduler states are ignored even if present.
                          Useful for transfer learning or fine-tuning.

    Returns:
        info dict with keys:
            epoch       (int)   — epoch index the checkpoint was saved at (0-based)
            loss        (float) — best test loss at that epoch
            results     (dict)  — full train/test loss (+ metric) history up to that epoch

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
        RuntimeError:      if the state_dict does not match the model architecture.

    Example — inference only:
        info = load_checkpoint("checkpoints/best.pth", model)
        model.eval()

    Example — resume training:
        info = load_checkpoint("checkpoints/best.pth", model, optimizer, scheduler)
        # then call training_loop() again starting from info["epoch"] + 1
    """

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    # Resolve target device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device)

    # model
    model.load_state_dict(checkpoint["model_state_dict"])

    # optimizer
    if not weights_only:
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    #  summary
    info = {
        "epoch":   checkpoint.get("epoch", -1),
        "loss":    checkpoint.get("loss",   float("inf")),
        "results": checkpoint.get("results", {}),
    }

    epoch_display = info["epoch"] + 1   
    mode = "weights only" if weights_only else "full state (model + optimizer + scheduler)"
    print(
        f"[INFO] Checkpoint loaded from '{path}'\n"
        f"       Saved at epoch : {epoch_display}\n"
        f"       Best test loss : {info['loss']:.4f}\n"
        f"       Restore mode   : {mode}"
    )

    return info

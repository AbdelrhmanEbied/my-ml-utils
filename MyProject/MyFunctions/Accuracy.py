import torch

def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes classification accuracy for both binary and multiclass problems.

    Binary:     y_pred shape (N,) or (N, 1) — raw logits or sigmoid outputs.
    Multiclass: y_pred shape (N, C)          — raw logits or softmax outputs.
    """
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        predicted_classes = (y_pred.squeeze() > 0.5).long()
    else:
        predicted_classes = torch.argmax(y_pred, dim=1)

    return (predicted_classes == y_true).float().mean().item()
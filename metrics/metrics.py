"""
metrics.py
==========
PyTorch-based evaluation metrics for binary and multiclass classification.

All functions operate on CPU or CUDA tensors and have **no external
dependencies** beyond ``torch``.

Conventions
-----------
* ``preds``  – raw model output (logits *or* probabilities/sigmoid).
* ``targets`` – ground-truth integer labels, shape ``(N,)``.
* ``is_logit`` – when ``True`` the decision boundary for binary
  classification is ``0``; when ``False`` (sigmoid / softmax output) it
  is ``0.5`` / ``argmax``.
* Division-by-zero is handled by returning ``0.0`` for the affected term.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

AveragingMode = Literal["macro", "micro", "weighted"]


def _to_predicted_labels(
    preds: Tensor,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> Tensor:
    """Convert raw model output to integer predicted labels.

    Parameters
    ----------
    preds:
        * **Binary** – shape ``(N,)`` or ``(N, 1)``.
        * **Multiclass** – shape ``(N, C)`` where *C* is the number of
          classes.
    is_logit:
        ``True``  → binary threshold is ``0``; multiclass uses ``argmax``.
        ``False`` → binary threshold is *threshold* (default ``0.5``);
        multiclass still uses ``argmax`` (works with softmax outputs too).
    threshold:
        Decision boundary used for binary sigmoid outputs when
        ``is_logit=False``.

    Returns
    -------
    Tensor
        Integer label tensor of shape ``(N,)``.
    """
    preds = preds.squeeze()  # (N,1) → (N,)

    if preds.dim() == 1:
        # ---- Binary --------------------------------------------------------
        if is_logit:
            return (preds >= 0.0).long()
        return (preds >= threshold).long()

    # ---- Multiclass --------------------------------------------------------
    return preds.argmax(dim=1)


def _safe_div(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Element-wise division; returns ``0.0`` wherever *denominator* is 0."""
    zero = torch.zeros_like(numerator, dtype=torch.float64)
    return torch.where(denominator == 0, zero, numerator / denominator)


def _confusion_matrix_raw(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """Return a ``(num_classes, num_classes)`` confusion matrix."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def accuracy(
    preds: Tensor,
    targets: Tensor,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> float:
    """Compute classification accuracy.

    .. math::

        \\text{Accuracy} = \\frac{\\text{Number of correct predictions}}
                                  {\\text{Total predictions}}

    For **binary** inputs the decision boundary is:

    * ``is_logit=True``  → threshold ``0``   (raw logits).
    * ``is_logit=False`` → threshold ``0.5`` (sigmoid outputs).

    For **multiclass** inputs ``argmax`` is always used.

    Parameters
    ----------
    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    is_logit:
        Whether *preds* are raw logits (binary only).
    threshold:
        Custom threshold for binary sigmoid outputs (ignored for logits).

    Returns
    -------
    float
        Accuracy in *[0, 1]*.
    """
    predicted = _to_predicted_labels(preds, is_logit=is_logit, threshold=threshold)
    correct = (predicted == targets.long()).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: Optional[int] = None,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> Tensor:
    """Return a confusion matrix as a ``torch.Tensor``.

    Rows correspond to **true** labels; columns to **predicted** labels:

    .. math::

        C_{i,j} = \\text{number of samples with true label } i
                  \\text{ predicted as label } j

    Parameters
    ----------
    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    num_classes:
        Total number of classes.  Inferred from *targets* when ``None``.
    is_logit:
        Whether *preds* are raw logits.
    threshold:
        Decision boundary for binary sigmoid outputs.

    Returns
    -------
    Tensor
        Long tensor of shape ``(num_classes, num_classes)``.
    """
    predicted = _to_predicted_labels(preds, is_logit=is_logit, threshold=threshold)
    if num_classes is None:
        num_classes = int(max(targets.max().item(), predicted.max().item())) + 1
    return _confusion_matrix_raw(predicted, targets, num_classes)


def precision(
    preds: Tensor,
    targets: Tensor,
    average: AveragingMode = "macro",
    num_classes: Optional[int] = None,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> float:
    r"""Compute precision.

    .. math::

        P = \frac{TP}{TP + FP}

    **Averaging modes** (multiclass):

    * ``"macro"``    – unweighted mean of per-class precision.
    * ``"micro"``    – global :math:`TP` and :math:`FP` summed across classes.
    * ``"weighted"`` – per-class precision weighted by support (true positives
      per class).

    Parameters
    ----------
    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    average:
        Averaging strategy – ``"macro"``, ``"micro"``, or ``"weighted"``.
    num_classes:
        Total number of classes.  Inferred from *targets* when ``None``.
    is_logit:
        Whether *preds* are raw logits.
    threshold:
        Decision boundary for binary sigmoid outputs.

    Returns
    -------
    float
        Precision value in *[0, 1]*.
    """
    predicted = _to_predicted_labels(preds, is_logit=is_logit, threshold=threshold)
    if num_classes is None:
        num_classes = int(max(targets.max().item(), predicted.max().item())) + 1

    cm = _confusion_matrix_raw(predicted, targets, num_classes).double()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp  # column sums minus diagonal
    support = cm.sum(dim=1)  # row sums = true counts per class

    if average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        return _safe_div(tp_sum, tp_sum + fp_sum).item()

    per_class = _safe_div(tp, tp + fp)

    if average == "macro":
        return per_class.mean().item()

    # weighted
    weights = support.double()
    total = weights.sum()
    if total == 0:
        return 0.0
    return (per_class * weights / total).sum().item()


def recall(
    preds: Tensor,
    targets: Tensor,
    average: AveragingMode = "macro",
    num_classes: Optional[int] = None,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> float:
    r"""Compute recall (sensitivity / true positive rate).

    .. math::

        R = \frac{TP}{TP + FN}

    **Averaging modes** (multiclass):

    * ``"macro"``    – unweighted mean of per-class recall.
    * ``"micro"``    – global :math:`TP` and :math:`FN` summed across classes.
    * ``"weighted"`` – per-class recall weighted by support.

    Parameters
    ----------
    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    average:
        Averaging strategy – ``"macro"``, ``"micro"``, or ``"weighted"``.
    num_classes:
        Total number of classes.  Inferred from *targets* when ``None``.
    is_logit:
        Whether *preds* are raw logits.
    threshold:
        Decision boundary for binary sigmoid outputs.

    Returns
    -------
    float
        Recall value in *[0, 1]*.
    """
    predicted = _to_predicted_labels(preds, is_logit=is_logit, threshold=threshold)
    if num_classes is None:
        num_classes = int(max(targets.max().item(), predicted.max().item())) + 1

    cm = _confusion_matrix_raw(predicted, targets, num_classes).double()
    tp = cm.diag()
    fn = cm.sum(dim=1) - tp  # row sums minus diagonal
    support = cm.sum(dim=1)

    if average == "micro":
        tp_sum = tp.sum()
        fn_sum = fn.sum()
        return _safe_div(tp_sum, tp_sum + fn_sum).item()

    per_class = _safe_div(tp, tp + fn)

    if average == "macro":
        return per_class.mean().item()

    # weighted
    weights = support.double()
    total = weights.sum()
    if total == 0:
        return 0.0
    return (per_class * weights / total).sum().item()


def f1_score(
    preds: Tensor,
    targets: Tensor,
    average: AveragingMode = "macro",
    num_classes: Optional[int] = None,
    is_logit: bool = True,
    threshold: float = 0.5,
) -> float:
    r"""Compute the F1 score (harmonic mean of precision and recall).

    .. math::

        F_1 = \frac{2 \cdot P \cdot R}{P + R}
            = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}

    **Averaging modes** (multiclass):

    * ``"macro"``    – unweighted mean of per-class F1.
    * ``"micro"``    – global counts summed across classes before computing F1.
    * ``"weighted"`` – per-class F1 weighted by support.

    Parameters
    ----------
    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    average:
        Averaging strategy – ``"macro"``, ``"micro"``, or ``"weighted"``.
    num_classes:
        Total number of classes.  Inferred from *targets* when ``None``.
    is_logit:
        Whether *preds* are raw logits.
    threshold:
        Decision boundary for binary sigmoid outputs.

    Returns
    -------
    float
        F1 score in *[0, 1]*.
    """
    predicted = _to_predicted_labels(preds, is_logit=is_logit, threshold=threshold)
    if num_classes is None:
        num_classes = int(max(targets.max().item(), predicted.max().item())) + 1

    cm = _confusion_matrix_raw(predicted, targets, num_classes).double()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)

    if average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        denom = 2 * tp_sum + fp_sum + fn_sum
        return _safe_div(2 * tp_sum, denom).item()

    per_class = _safe_div(2 * tp, 2 * tp + fp + fn)

    if average == "macro":
        return per_class.mean().item()

    # weighted
    weights = support.double()
    total = weights.sum()
    if total == 0:
        return 0.0
    return (per_class * weights / total).sum().item()


# ---------------------------------------------------------------------------
# Metric dispatcher
# ---------------------------------------------------------------------------

#: All metric names recognised by :func:`calculate_metrics`.
AVAILABLE_METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
]


def calculate_metrics(
    metric_names: List[str],
    preds: Tensor,
    targets: Tensor,
    *,
    is_logit: bool = True,
    threshold: float = 0.5,
    average: AveragingMode = "macro",
    num_classes: Optional[int] = None,
) -> Dict[str, object]:
    """Compute a requested subset of evaluation metrics.

    Parameters
    ----------
    metric_names:
        List of metric names to compute.  Supported values:

        * ``"accuracy"``
        * ``"precision"``
        * ``"recall"``
        * ``"f1_score"``
        * ``"confusion_matrix"``

    preds:
        Model output – shape ``(N,)`` / ``(N, 1)`` for binary or
        ``(N, C)`` for multiclass.
    targets:
        Ground-truth integer labels – shape ``(N,)``.
    is_logit:
        Whether *preds* are raw logits.  Forwarded to every metric.
    threshold:
        Decision boundary for binary sigmoid outputs.
    average:
        Averaging strategy for precision / recall / F1 –
        ``"macro"``, ``"micro"``, or ``"weighted"``.
    num_classes:
        Total number of classes.  Inferred from *targets* when ``None``.

    Returns
    -------
    dict
        ``{metric_name: value}`` for each requested metric.

    Raises
    ------
    ValueError
        If an unrecognised metric name is supplied.

    Examples
    --------
    >>> import torch
    >>> logits = torch.tensor([2.0, -1.0, 0.5, -0.3])
    >>> targets = torch.tensor([1, 0, 1, 0])
    >>> results = calculate_metrics(
    ...     ["accuracy", "f1_score"],
    ...     logits, targets,
    ...     is_logit=True,
    ... )
    >>> print(results)
    {'accuracy': 1.0, 'f1_score': 1.0}
    """
    unknown = set(metric_names) - set(AVAILABLE_METRICS)
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {unknown}.  "
            f"Available metrics: {AVAILABLE_METRICS}"
        )

    shared_kwargs = dict(
        preds=preds,
        targets=targets,
        is_logit=is_logit,
        threshold=threshold,
    )
    averaging_kwargs = dict(**shared_kwargs, average=average, num_classes=num_classes)

    _dispatch = {
        "accuracy":         lambda: accuracy(**shared_kwargs),
        "precision":        lambda: precision(**averaging_kwargs),
        "recall":           lambda: recall(**averaging_kwargs),
        "f1_score":         lambda: f1_score(**averaging_kwargs),
        "confusion_matrix": lambda: confusion_matrix(
            **shared_kwargs, num_classes=num_classes
        ),
    }

    return {name: _dispatch[name]() for name in metric_names}


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Binary logit example ───────────────────────────────────────────────
    print("=" * 60)
    print("Binary classification (logits)")
    print("=" * 60)
    bin_logits  = torch.tensor([ 2.1, -0.5,  1.3, -1.8,  0.7, -0.2])
    bin_targets = torch.tensor([   1,    0,    1,    0,    1,    1 ])

    bin_results = calculate_metrics(
        ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"],
        bin_logits,
        bin_targets,
        is_logit=True,
        average="macro",
    )
    for k, v in bin_results.items():
        print(f"  {k:20s}: {v}")

    # ── Multiclass softmax example ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("Multiclass classification (softmax, 3 classes)")
    print("=" * 60)
    mc_probs = torch.softmax(
        torch.tensor([
            [3.0, 1.0, 0.5],
            [0.2, 2.5, 0.3],
            [0.1, 0.4, 3.1],
            [2.8, 0.3, 0.2],
            [0.3, 0.2, 2.9],
        ]),
        dim=1,
    )
    mc_targets = torch.tensor([0, 1, 2, 0, 1])  # last sample is wrong

    for avg in ("macro", "micro", "weighted"):
        mc_results = calculate_metrics(
            ["accuracy", "precision", "recall", "f1_score"],
            mc_probs,
            mc_targets,
            is_logit=False,
            average=avg,        # type: ignore[arg-type]
            num_classes=3,
        )
        print(f"\n  average='{avg}'")
        for k, v in mc_results.items():
            print(f"    {k:20s}: {v:.4f}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Confusion matrix (multiclass)")
    print("=" * 60)
    cm = confusion_matrix(mc_probs, mc_targets, num_classes=3, is_logit=False)
    print(f"\n  CM (rows=true, cols=pred):\n{cm}\n")

    # ── Edge-case: all predictions for one class ───────────────────────────
    print("=" * 60)
    print("Edge case: zero division guard")
    print("=" * 60)
    edge_preds   = torch.tensor([0, 0, 0, 0])   # always predicts class 0
    edge_targets = torch.tensor([0, 1, 0, 1])
    edge_results = calculate_metrics(
        ["precision", "recall", "f1_score"],
        edge_preds,
        edge_targets,
        is_logit=False,          # already label integers, threshold irrelevant
        average="macro",
        num_classes=2,
    )
    for k, v in edge_results.items():
        print(f"  {k:20s}: {v:.4f}")

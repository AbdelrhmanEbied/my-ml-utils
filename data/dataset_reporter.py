import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm.auto import tqdm


def analyze_dataset_report(
    dataset,
    expected_shape: tuple,
    batch_size: int = 64,
    num_workers: int = 4,
    class_names: dict = None,
) -> dict:
    """
    Performs a full audit of a PyTorch image dataset and returns a structured
    report dictionary alongside a formatted printout.

    Args:
        dataset:        A PyTorch Dataset that returns (image_tensor, label).
        expected_shape: The expected (C, H, W) for every image in the dataset.
        batch_size:     DataLoader batch size. Default: 64.
        num_workers:    DataLoader worker count. Default: 4.
        class_names:    Optional dict mapping class ID -> human-readable name.
                        e.g. {0: "cat", 1: "dog"}. Falls back to "Class [ID]" if None.

    Returns:
        report (dict): Structured audit report with the following keys:
            - "normalization":      {"mean": [...], "std": [...]}
            - "value_range":        {"global_min": float, "global_max": float}
            - "class_distribution": {class_label: count, ...}
            - "imbalance_ratio":    float
            - "resolution_audit":   {"expected": tuple, "status": str, "outliers": list}
            - "dataset_size":       int
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = expected_shape

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Global Sum method — float64 for precision stability
    total_sum            = torch.zeros(C, dtype=torch.float64, device=device)
    total_sum_of_squares = torch.zeros(C, dtype=torch.float64, device=device)
    total_pixels         = 0

    global_min    = torch.tensor(float("inf"),  device=device)
    global_max    = torch.tensor(float("-inf"), device=device)
    class_counter = Counter()
    outliers      = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Auditing dataset")):

        # Keep stats in float32/float64 — explicitly disable autocast influence
        with torch.amp.autocast(device_type=device.type, enabled=False):
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)

            b, c, h, w = images.shape

            # Resolution audit — compare individual image shape (C, H, W), not batch (B, C, H, W)
            if (c, h, w) != expected_shape:
                outliers.append({
                    "batch_index":    batch_idx,
                    "found_shape":    (c, h, w),
                    "expected_shape": expected_shape,
                })

            # Global Sum accumulation in float64 to prevent precision drift
            images_f64            = images.double()
            total_sum            += images_f64.sum(dim=[0, 2, 3])
            total_sum_of_squares += (images_f64 ** 2).sum(dim=[0, 2, 3])
            total_pixels         += b * h * w

            # Value range
            global_min = torch.minimum(global_min, images.min())
            global_max = torch.maximum(global_max, images.max())

        # Class distribution
        if isinstance(labels, torch.Tensor):
            class_counter.update(labels.tolist())
        else:
            class_counter.update(labels)

    # Final normalization — single division at the end, no drift accumulation
    mean = (total_sum / total_pixels).float().cpu()
    std  = ((total_sum_of_squares / total_pixels) - (total_sum / total_pixels) ** 2).sqrt().float().cpu()

    def r(val):
        return round(float(val), 4)

    # Class distribution — use class_names if provided, else "Class [ID]"
    class_distribution = {}
    for class_id, count in sorted(class_counter.items()):
        if class_names and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        class_distribution[label] = count

    total_images    = len(dataset)
    max_count       = max(class_distribution.values())
    min_count       = min(class_distribution.values())
    imbalance_ratio = round(max_count / max(min_count, 1), 4)

    report = {
        "dataset_size": total_images,
        "normalization": {
            "mean": [r(m) for m in mean],
            "std":  [r(s) for s in std],
        },
        "value_range": {
            "global_min": r(global_min),
            "global_max": r(global_max),
        },
        "class_distribution": class_distribution,
        "imbalance_ratio":    imbalance_ratio,
        "resolution_audit": {
            "expected": expected_shape,
            "status":   "PASS" if not outliers else f"FAIL — {len(outliers)} mismatched batch(es)",
            "outliers": outliers,
        },
    }

    _print_report(report)
    return report


def _print_report(report: dict) -> None:

    W    = 62
    sep  = "═" * W
    thin = "─" * W

    vmin       = report["value_range"]["global_min"]
    vmax       = report["value_range"]["global_max"]
    range_ok   = vmax <= 1.0
    res_ok     = report["resolution_audit"]["status"] == "PASS"
    balance_ok = report["imbalance_ratio"] < 5.0

    def badge(ok):
        return "✓ PASS" if ok else "✗ FAIL"

    # Header
    print(f"\n{sep}")
    print(f"{'DATASET AUDIT REPORT':^{W}}")
    print(f"{sep}")
    print(f"  Total Images : {report['dataset_size']:,}")

    # Data Health Summary
    print(f"\n{'DATA HEALTH SUMMARY':^{W}}")
    print(thin)
    print(f"  {'Check':<30} {'Result':>20}")
    print(thin)
    print(f"  {'Pixel Range (0–1)':<30} {badge(range_ok):>20}")
    print(f"  {'Resolution Consistency':<30} {badge(res_ok):>20}")
    print(f"  {'Class Balance (ratio < 5)':<30} {badge(balance_ok):>20}")
    overall = range_ok and res_ok and balance_ok
    print(thin)
    print(f"  {'Overall Health':<30} {badge(overall):>20}")

    # Normalization Stats
    print(f"\n{'NORMALIZATION  (per-channel)':^{W}}")
    print(thin)
    print(f"  {'Channel':<12} {'Mean':>12} {'Std':>12}")
    print(thin)
    mean = report["normalization"]["mean"]
    std  = report["normalization"]["std"]
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  {'Ch ' + str(i):<12} {m:>12.4f} {s:>12.4f}")

    # Value Range Audit
    print(f"\n{'VALUE RANGE AUDIT':^{W}}")
    print(thin)
    scale = "0–1  ✓  normalised" if range_ok else "0–255  ✗  raw uint8 — normalise before training"
    print(f"  {'Global Min':<20} {vmin:>10.4f}")
    print(f"  {'Global Max':<20} {vmax:>10.4f}")
    print(f"  {'Scale Detected':<20} {scale}")

    # Class Distribution
    print(f"\n{'CLASS DISTRIBUTION':^{W}}")
    print(thin)
    total = report["dataset_size"]
    print(f"  {'Class':<18} {'Count':>7} {'%':>7}  {'Bar'}")
    print(thin)
    for label, count in report["class_distribution"].items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {str(label):<18} {count:>7,} {pct:>6.1f}%  {bar}")
    print(thin)
    print(f"  {'Imbalance Ratio':<18} {report['imbalance_ratio']:>7.4f}  (max class / min class)")
    if not balance_ok:
        print(f"  ⚠  Ratio > 5 — consider oversampling, class weights, or augmentation.")

    # Resolution Audit
    print(f"\n{'RESOLUTION AUDIT':^{W}}")
    print(thin)
    audit = report["resolution_audit"]
    print(f"  {'Expected Shape':<24} {str(audit['expected']):>12}")
    print(f"  {'Status':<24} {audit['status']:>12}")
    if audit["outliers"]:
        print(thin)
        print(f"  {'Batch Idx':<12} {'Found Shape':>20} {'Expected':>20}")
        print(thin)
        for o in audit["outliers"]:
            print(f"  {o['batch_index']:<12} {str(o['found_shape']):>20} {str(o['expected_shape']):>20}")

    print(f"\n{sep}\n")

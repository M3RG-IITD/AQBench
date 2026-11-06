import numpy as np


def compute_constraint_violations(preds_phys, target_features):
    """
    Computes physical constraint violation rates on denormalised predictions.

    Args:
        preds_phys:      (N, H, W, T_out, num_targets) float32 — denormalised
        target_features: list of feature name strings

    Returns:
        dict of violation rate scalars (values in [0,1])
    """
    violations = {}

    for i, feat in enumerate(target_features):
        rate = float((preds_phys[..., i] < 0).mean())
        violations[f"{feat}_negative_rate"] = rate

    # pm10 >= pm25 everywhere
    if "pm10" in target_features and "pm25" in target_features:
        pm25_i = target_features.index("pm25")
        pm10_i = target_features.index("pm10")
        rate   = float((preds_phys[..., pm10_i] < preds_phys[..., pm25_i]).mean())
        violations["pm10_lt_pm25_rate"] = rate

    return violations
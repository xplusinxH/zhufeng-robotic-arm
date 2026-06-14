"""Depth validation and filtering helpers."""


def is_valid_depth(depth_m: float, min_m: float = 0.15, max_m: float = 1.20) -> bool:
    """Return whether a depth value is inside the configured working range."""
    return min_m <= depth_m <= max_m

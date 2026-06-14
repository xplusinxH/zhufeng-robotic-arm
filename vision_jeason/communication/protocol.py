"""ASCII serial protocol formatting."""


def format_no_target() -> str:
    """Return the no-target protocol frame."""
    return "@NO_TARGET#"


def format_target(class_name: str, x_mm: float, y_mm: float, z_mm: float, score: float) -> str:
    """Return a single-target protocol frame."""
    return f"@TARGET,{class_name},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},{score:.2f}#"

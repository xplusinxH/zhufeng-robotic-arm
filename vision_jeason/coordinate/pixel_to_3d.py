"""Convert aligned pixel and depth values to camera coordinates."""


def pixel_to_camera(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Convert a pixel coordinate and depth to camera-frame XYZ in meters."""
    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    return x_m, y_m, z_m

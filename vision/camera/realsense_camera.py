"""RealSense D435 capture wrapper."""


class RealSenseCamera:
    """Placeholder interface for the Jetson RealSense capture implementation."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps

    def describe(self) -> dict[str, int]:
        """Return the configured stream shape."""
        return {"width": self.width, "height": self.height, "fps": self.fps}

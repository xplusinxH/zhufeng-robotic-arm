from pathlib import Path

from camera.depth_utils import is_valid_depth
from camera.realsense_camera import RealSenseCamera
from communication.protocol import format_no_target, format_target
from coordinate.pixel_to_3d import pixel_to_camera


def test_config_file_exists() -> None:
    assert Path("config.yaml").is_file()


def test_camera_description() -> None:
    camera = RealSenseCamera()
    assert camera.describe() == {"width": 640, "height": 480, "fps": 30}


def test_depth_range_validation() -> None:
    assert is_valid_depth(0.30)
    assert not is_valid_depth(0.0)
    assert not is_valid_depth(2.0)


def test_pixel_to_camera_center() -> None:
    assert pixel_to_camera(320, 240, 0.5, 615, 615, 320, 240) == (0.0, 0.0, 0.5)


def test_serial_protocol_frames() -> None:
    assert format_no_target() == "@NO_TARGET#"
    assert format_target("bottle", 126.4, -35.2, 280.7, 0.86) == "@TARGET,bottle,126.4,-35.2,280.7,0.86#"

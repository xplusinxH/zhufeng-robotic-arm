import ast
from pathlib import Path
import unittest

from camera.realsense_camera import RealSenseCamera


class FakeConfig:
    def __init__(self):
        self.streams = []

    def enable_stream(self, *args):
        self.streams.append(args)


class FakePipeline:
    def __init__(self, frames):
        self.frames = frames
        self.started_with = None
        self.stopped = False

    def start(self, config):
        self.started_with = config

    def wait_for_frames(self):
        return self.frames

    def stop(self):
        self.stopped = True


class FakeAlign:
    def __init__(self, aligned_frames):
        self.aligned_frames = aligned_frames
        self.received = None

    def process(self, frames):
        self.received = frames
        return self.aligned_frames


class FakeFrames:
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class FakeRealSense:
    class stream:
        color = "color"
        depth = "depth"

    class format:
        bgr8 = "bgr8"
        z16 = "z16"

    def __init__(self):
        self.raw_frames = object()
        self.color_frame = object()
        self.depth_frame = object()
        self.aligned_frames = FakeFrames(self.color_frame, self.depth_frame)
        self.pipeline_instance = FakePipeline(self.raw_frames)
        self.config_instance = FakeConfig()
        self.align_instance = FakeAlign(self.aligned_frames)

    def pipeline(self):
        return self.pipeline_instance

    def config(self):
        return self.config_instance

    def align(self, stream):
        assert stream == self.stream.color
        return self.align_instance


class RealSenseCameraTests(unittest.TestCase):
    def test_camera_starts_color_and_depth_streams(self):
        rs = FakeRealSense()
        camera = RealSenseCamera(rs_module=rs)

        camera.start()

        self.assertEqual(
            rs.config_instance.streams,
            [
                ("depth", 640, 480, "z16", 30),
                ("color", 640, 480, "bgr8", 30),
            ],
        )
        self.assertIs(rs.pipeline_instance.started_with, rs.config_instance)

    def test_camera_returns_aligned_color_and_depth_frames(self):
        rs = FakeRealSense()
        camera = RealSenseCamera(rs_module=rs)
        camera.start()

        color, depth = camera.capture_aligned()

        self.assertIs(color, rs.color_frame)
        self.assertIs(depth, rs.depth_frame)
        self.assertIs(rs.align_instance.received, rs.raw_frames)

    def test_camera_rejects_capture_before_start(self):
        camera = RealSenseCamera(rs_module=FakeRealSense())

        with self.assertRaisesRegex(RuntimeError, "尚未启动"):
            camera.capture_aligned()

    def test_camera_stops_pipeline(self):
        rs = FakeRealSense()
        camera = RealSenseCamera(rs_module=rs)
        camera.start()

        camera.stop()

        self.assertTrue(rs.pipeline_instance.stopped)

    def test_camera_module_syntax_is_compatible_with_python_36(self):
        source = Path("camera/realsense_camera.py").read_text(encoding="utf-8")

        ast.parse(source, feature_version=(3, 6))


if __name__ == "__main__":
    unittest.main()

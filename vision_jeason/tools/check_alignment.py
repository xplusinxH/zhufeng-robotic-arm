"""在 Jetson 屏幕上实时验证 D435 RGB 与深度对齐。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.realsense_camera import RealSenseCamera
from coordinate.pixel_to_3d import pixel_depth_to_camera
from tools.alignment_data import create_capture_directory, save_measurement


WINDOW_NAME = "D435 Alignment Check"
DEFAULT_OUTPUT_ROOT = Path("/mnt/zhufeng_data/zhufeng/data/alignment")


def intrinsics_to_dict(intrinsics):
    return {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
    }


def make_depth_colormap(depth_raw, depth_scale, min_depth_m, max_depth_m):
    depth_m = depth_raw.astype(np.float32) * depth_scale
    normalized = (depth_m - min_depth_m) * 255.0 / (max_depth_m - min_depth_m)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    normalized[depth_raw == 0] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def draw_measurement(image, pixel, depth_m, camera_point):
    if pixel is None:
        return image
    output = image.copy()
    cv2.drawMarker(output, pixel, (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    if camera_point is None:
        text = "u={} v={} invalid depth".format(pixel[0], pixel[1])
    else:
        text = "u={} v={} Z={:.3f}m XYZ=({:.3f},{:.3f},{:.3f})m".format(
            pixel[0],
            pixel[1],
            depth_m,
            camera_point[0],
            camera_point[1],
            camera_point[2],
        )
    cv2.putText(output, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return output


def save_capture(
    output_root,
    color_image,
    depth_raw,
    depth_colormap,
    overlay,
    pixel,
    depth_m,
    camera_point,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    capture_dir = create_capture_directory(output_root, timestamp)
    cv2.imwrite(str(capture_dir / "color.png"), color_image)
    cv2.imwrite(str(capture_dir / "depth_raw.png"), depth_raw)
    cv2.imwrite(str(capture_dir / "depth_colormap.png"), depth_colormap)
    cv2.imwrite(str(capture_dir / "overlay.png"), overlay)
    save_measurement(
        capture_dir / "measurement.json",
        pixel if pixel is not None else (-1, -1),
        depth_m,
        camera_point,
    )
    return capture_dir


def run(output_root, min_depth_m, max_depth_m):
    camera = RealSenseCamera()
    selected_pixel = [None]
    latest = {}

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and x < camera.width and y < camera.height:
            selected_pixel[0] = (x, y)

    camera.start()
    intrinsics = intrinsics_to_dict(camera.get_aligned_depth_intrinsics())
    depth_scale = camera.get_depth_scale()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            color_frame, depth_frame = camera.capture_aligned()
            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_colormap = make_depth_colormap(
                depth_raw, depth_scale, min_depth_m, max_depth_m
            )

            pixel = selected_pixel[0]
            depth_m = 0.0
            camera_point = None
            if pixel is not None:
                depth_m = depth_frame.get_distance(pixel[0], pixel[1])
                camera_point = pixel_depth_to_camera(
                    pixel[0], pixel[1], depth_m, intrinsics
                )

            annotated_color = draw_measurement(
                color_image, pixel, depth_m, camera_point
            )
            annotated_depth = draw_measurement(
                depth_colormap, pixel, depth_m, camera_point
            )
            overlay = cv2.addWeighted(color_image, 0.6, depth_colormap, 0.4, 0)
            combined = np.hstack((annotated_color, annotated_depth))
            cv2.imshow(WINDOW_NAME, combined)

            latest = {
                "color": color_image,
                "depth_raw": depth_raw,
                "depth_colormap": depth_colormap,
                "overlay": overlay,
                "pixel": pixel,
                "depth_m": depth_m,
                "camera_point": camera_point,
            }

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                capture_dir = save_capture(
                    output_root,
                    latest["color"],
                    latest["depth_raw"],
                    latest["depth_colormap"],
                    latest["overlay"],
                    latest["pixel"],
                    latest["depth_m"],
                    latest["camera_point"],
                )
                print("验证数据已保存到：{}".format(capture_dir))
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=1.20)
    args = parser.parse_args()
    if args.max_depth <= args.min_depth:
        parser.error("--max-depth 必须大于 --min-depth")
    run(args.output_root, args.min_depth, args.max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


EXPECTED_PROTOCOL = "sukinee_tag_pose_v1"
REQUEST_PROTOCOL = "sukinee_sample_request_v1"
DEFAULT_SERIAL_REQUEST_TEXT = "@GET_TAG_POSE#"


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def as_float_list(value: Any, n: int, field_name: str) -> List[float]:
    if isinstance(value, dict):
        keys = ["x", "y", "z"] if n == 3 else ["x", "y", "z", "w"]
        try:
            value = [value[k] for k in keys]
        except KeyError as exc:
            raise ValueError(f"{field_name} dict missing key {exc!s}") from exc

    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{field_name} must be a list length {n} or xyz/xyzw dict")

    out = []
    for idx, item in enumerate(value):
        if not is_finite_number(item):
            raise ValueError(f"{field_name}[{idx}] is not a finite number: {item!r}")
        out.append(float(item))
    return out


def get_nested(msg: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = msg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_position_orientation(msg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    pos = msg.get("position_m")
    quat = msg.get("orientation_xyzw")

    if pos is None:
        pos = get_nested(msg, "T_R_T", "position")
    if quat is None:
        quat = get_nested(msg, "T_R_T", "orientation_xyzw")

    if pos is None:
        pos = get_nested(msg, "transform", "position")
    if quat is None:
        quat = get_nested(msg, "transform", "orientation_xyzw")

    if pos is None:
        raise ValueError("missing position_m or T_R_T.position")
    if quat is None:
        raise ValueError("missing orientation_xyzw or T_R_T.orientation_xyzw")

    return as_float_list(pos, 3, "position_m"), as_float_list(quat, 4, "orientation_xyzw")


def quat_norm(q: Iterable[float]) -> float:
    q_list = list(q)
    return math.sqrt(sum(v * v for v in q_list))


def validate_and_canonicalize(
    msg: Dict[str, Any],
    *,
    expected_family: str,
    expected_base_id: int,
    expected_tool_id: int,
    expected_from_frame: str,
    expected_to_frame: str,
    quat_norm_tolerance: float,
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not isinstance(msg, dict):
        raise ValueError("packet is not a JSON object")

    sample_id = msg.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id is required and must be a non-empty string")
    sample_id = sample_id.strip()

    protocol = msg.get("protocol", EXPECTED_PROTOCOL)
    if protocol != EXPECTED_PROTOCOL:
        warnings.append(f"protocol is {protocol!r}, expected {EXPECTED_PROTOCOL!r}")

    if msg.get("valid") is False:
        raise ValueError(f"vision side marked packet invalid: {msg.get('reject_reason', '')}")

    tag_family = msg.get("tag_family", expected_family)
    if tag_family != expected_family:
        raise ValueError(f"tag_family={tag_family!r}, expected {expected_family!r}")

    base_id = msg.get("tag_base_ref_id", expected_base_id)
    tool_id = msg.get("tag_tool0_id", expected_tool_id)

    if int(base_id) != expected_base_id:
        raise ValueError(f"tag_base_ref_id={base_id}, expected {expected_base_id}")
    if int(tool_id) != expected_tool_id:
        raise ValueError(f"tag_tool0_id={tool_id}, expected {expected_tool_id}")

    from_frame = msg.get("from_frame", expected_from_frame)
    to_frame = msg.get("to_frame", expected_to_frame)

    if from_frame != expected_from_frame:
        raise ValueError(f"from_frame={from_frame!r}, expected {expected_from_frame!r}")
    if to_frame != expected_to_frame:
        raise ValueError(f"to_frame={to_frame!r}, expected {expected_to_frame!r}")

    position_m, orientation_xyzw = extract_position_orientation(msg)

    qn = quat_norm(orientation_xyzw)
    if abs(qn - 1.0) > quat_norm_tolerance:
        raise ValueError(
            f"quaternion norm {qn:.6f} exceeds tolerance {quat_norm_tolerance:.6f}; "
            "check xyzw order and normalization"
        )

    if abs(qn - 1.0) > 0.01:
        warnings.append(f"quaternion norm is {qn:.6f}, not close to 1")

    tag_size_m = msg.get("tag_size_m")
    if tag_size_m is not None:
        if not is_finite_number(tag_size_m) or float(tag_size_m) <= 0:
            raise ValueError(f"tag_size_m must be positive, got {tag_size_m!r}")
        tag_size_m = float(tag_size_m)

    relative_pose_source = msg.get("relative_pose_source", "unknown")
    if relative_pose_source not in {"same_frame", "cached_base_ref", "unknown"}:
        warnings.append(f"unknown relative_pose_source={relative_pose_source!r}")

    quality = msg.get("quality", {})
    if quality is None:
        quality = {}
    if not isinstance(quality, dict):
        warnings.append("quality field is not a dict; saved under raw only")
        quality = {}

    canonical = {
        "protocol": EXPECTED_PROTOCOL,
        "sample_id": sample_id,
        "timestamp_pc_receive": utc_now_iso(),
        "timestamp_jetson": msg.get("timestamp_jetson", msg.get("timestamp_vision")),
        "from_frame": expected_from_frame,
        "to_frame": expected_to_frame,
        "meaning": "tag_base_ref -> tag_tool0",
        "tag_family": expected_family,
        "tag_base_ref_id": expected_base_id,
        "tag_tool0_id": expected_tool_id,
        "tag_size_m": tag_size_m,
        "T_R_T": {
            "meaning": "tag_base_ref -> tag_tool0",
            "position": {
                "x": position_m[0],
                "y": position_m[1],
                "z": position_m[2]
            },
            "orientation_xyzw": {
                "x": orientation_xyzw[0],
                "y": orientation_xyzw[1],
                "z": orientation_xyzw[2],
                "w": orientation_xyzw[3]
            },
            "quaternion_norm": qn
        },
        "relative_pose_source": relative_pose_source,
        "base_ref_cache_age_sec": msg.get("base_ref_cache_age_sec"),
        "base_ref_cache_age_samples": msg.get("base_ref_cache_age_samples"),
        "frame_count_used": msg.get("frame_count_used"),
        "quality": quality,
        "valid": True,
        "validation_warnings": warnings,
        "raw": msg
    }

    # Normalize base_ref_source from either top-level field or quality.base_ref_source.
    # Vision-side protocol may place base_ref_source inside the quality dict.
    quality_for_source = canonical.get("quality", {})
    if not isinstance(quality_for_source, dict):
        quality_for_source = {}

    base_ref_source_norm = (
        canonical.get("base_ref_source")
        or msg.get("base_ref_source")
        or quality_for_source.get("base_ref_source")
    )

    if base_ref_source_norm:
        canonical["base_ref_source"] = base_ref_source_norm

        if canonical.get("relative_pose_source", "unknown") == "unknown":
            if base_ref_source_norm == "live":
                canonical["relative_pose_source"] = "same_frame"
            elif base_ref_source_norm == "cached":
                canonical["relative_pose_source"] = "cached_base_ref"
            elif base_ref_source_norm == "mixed":
                canonical["relative_pose_source"] = "mixed"
            else:
                canonical["relative_pose_source"] = "unknown"

    return canonical, warnings


class LineSource:
    def readline(self) -> bytes:
        raise NotImplementedError

    def write_line(self, line: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class StdinLineSource(LineSource):
    def readline(self) -> bytes:
        line = sys.stdin.readline()
        return line.encode("utf-8") if line else b""

    def write_line(self, line: str) -> None:
        print(f"[stdin-mode would-send] {line.rstrip()}", file=sys.stderr)


class SerialLineSource(LineSource):
    def __init__(self, port: str, baud: int, timeout: float):
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Install it with: python3 -m pip install pyserial"
            ) from exc

        self._serial = serial.Serial(port=port, baudrate=baud, timeout=timeout)

    def readline(self) -> bytes:
        return self._serial.readline()

    def write_line(self, line: str) -> None:
        data = line.encode("utf-8")
        self._serial.write(data)
        self._serial.flush()

    def close(self) -> None:
        self._serial.close()


def open_line_source(port: str, baud: int, timeout: float) -> LineSource:
    if port == "-":
        return StdinLineSource()
    return SerialLineSource(port, baud, timeout)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_line(raw_line: bytes, *, max_line_len: int) -> Dict[str, Any]:
    if not raw_line:
        raise TimeoutError("no data")
    if len(raw_line) > max_line_len:
        raise ValueError(f"line too long: {len(raw_line)} bytes > {max_line_len}")

    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text:
        raise TimeoutError("empty line")

    return json.loads(text)


def make_request(sample_id: str, expected_base_id: int, expected_tool_id: int) -> Dict[str, Any]:
    return {
        "protocol": REQUEST_PROTOCOL,
        "cmd": "capture_relative_pose",
        "sample_id": sample_id,
        "tag_family": "tag25h9",
        "tag_base_ref_id": expected_base_id,
        "tag_tool0_id": expected_tool_id,
        "from_frame": "tag_base_ref",
        "to_frame": "tag_tool0"
    }


def handle_packet(
    msg: Dict[str, Any],
    args: argparse.Namespace,
    *,
    required_sample_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    canonical, warnings = validate_and_canonicalize(
        msg,
        expected_family=args.tag_family,
        expected_base_id=args.base_id,
        expected_tool_id=args.tool_id,
        expected_from_frame=args.from_frame,
        expected_to_frame=args.to_frame,
        quat_norm_tolerance=args.quat_norm_tolerance,
    )

    if required_sample_id and canonical["sample_id"] != required_sample_id:
        print(
            f"[skip] got sample_id={canonical['sample_id']}, waiting for {required_sample_id}",
            file=sys.stderr,
        )
        return None

    append_jsonl(Path(args.out), canonical)

    warn_text = f" warnings={warnings}" if warnings else ""
    print(
        f"[saved] {canonical['sample_id']} -> {args.out} "
        f"source={canonical['relative_pose_source']} "
        f"qnorm={canonical['T_R_T']['quaternion_norm']:.6f}"
        f"{warn_text}",
        file=sys.stderr,
    )

    return canonical


def run_listen(args: argparse.Namespace) -> int:
    source = open_line_source(args.port, args.baud, args.timeout)

    print(f"[listen] port={args.port} baud={args.baud} out={args.out}", file=sys.stderr)

    try:
        while True:
            try:
                msg = parse_line(source.readline(), max_line_len=args.max_line_len)
                handle_packet(msg, args, required_sample_id=args.sample_id)
            except TimeoutError:
                continue
            except KeyboardInterrupt:
                print("\n[exit] interrupted", file=sys.stderr)
                return 0
            except Exception as exc:
                print(f"[reject] {exc}", file=sys.stderr)
                if args.fail_fast:
                    return 2
    finally:
        source.close()


def run_request_once(args: argparse.Namespace) -> int:
    """Send one vision request command, then wait for one valid JSONL response.

    Current vision-side protocol:
    PC sends plain text command: @GET_TAG_POSE#
    Jetson captures frames and returns one JSON line.
    """
    source = open_line_source(args.port, args.baud, args.timeout)

    req_line = getattr(args, "request_text", DEFAULT_SERIAL_REQUEST_TEXT)
    if not isinstance(req_line, str) or not req_line:
        req_line = DEFAULT_SERIAL_REQUEST_TEXT

    if not req_line.endswith("\n"):
        req_line += "\n"

    print(f"[request] {req_line.rstrip()}", file=sys.stderr)
    source.write_line(req_line)

    deadline = time.monotonic() + args.request_timeout

    try:
        while time.monotonic() < deadline:
            try:
                msg = parse_line(source.readline(), max_line_len=args.max_line_len)
                saved = handle_packet(msg, args, required_sample_id=args.sample_id)
                if saved is not None:
                    return 0
            except TimeoutError:
                continue
            except Exception as exc:
                print(f"[reject] {exc}", file=sys.stderr)
                if args.fail_fast:
                    return 2

        print("[timeout] no valid response from Jetson", file=sys.stderr)
        return 1

    finally:
        source.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receive Sukinee tag_base_ref -> tag_tool0 JSONL pose packets from Jetson serial."
    )

    p.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port, or '-' to read JSONL from stdin"
    )
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument(
        "--timeout",
        type=float,
        default=0.2,
        help="Serial readline timeout in seconds"
    )
    p.add_argument(
        "--out",
        default="/home/zzj/sukinee_ws/vision_calibration/data/vision_samples_trial.jsonl",
        help="Output JSONL path"
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="Only accept this sample_id; required with --request"
    )
    p.add_argument(
        "--request",
        action="store_true",
        help="Send one capture request then wait for matching sample_id"
    )
    p.add_argument("--request-timeout", type=float, default=10.0)
    p.add_argument("--request-text", default=DEFAULT_SERIAL_REQUEST_TEXT, help="Text command sent to Jetson in --request mode")
    p.add_argument("--tag-family", default="tag25h9")
    p.add_argument("--base-id", type=int, default=0, help="tag_base_ref id")
    p.add_argument("--tool-id", type=int, default=1, help="tag_tool0 id")
    p.add_argument("--from-frame", default="tag_base_ref")
    p.add_argument("--to-frame", default="tag_tool0")
    p.add_argument("--quat-norm-tolerance", type=float, default=0.05)
    p.add_argument("--max-line-len", type=int, default=20000)
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit on first invalid packet"
    )

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.request:
        return run_request_once(args)

    return run_listen(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Read Sukinee current q_urdf once from real motor feedback.

Purpose:
  - Read real motor position feedback through Type17 read-only parameter requests.
  - Convert motor_pos to Joint1~Joint6 q_urdf using software offset JSON.
  - Print COPY_THIS_CURRENT_Q for trajectory_safety_check.py / executor.
  - Write /tmp/sukinee_current_q.txt by default.

Safety boundary:
  - Opens SocketCAN only for read-only Type17 parameter reads.
  - Sends NO Type1 motion command.
  - Sends NO Type3 enable.
  - Sends NO Type4 disable.
  - Sends NO Type6 set zero.
  - Sends NO Type18 write parameter.
  - Saves NO motor parameter.
  - Changes NO CAN_ID.
  - Switches NO protocol.
  - Does NOT invoke MoveIt real execution.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[2]
DRIVER_DIR = SRC_DIR / "sukinee_urdf" / "scripts" / "zero_drag"

if not DRIVER_DIR.exists():
    raise RuntimeError(f"Cannot find Sukinee zero_drag driver dir: {DRIVER_DIR}")

if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

from sukinee_socketcan_driver import SukineeSocketCANDriver  # noqa: E402


ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]

JOINT_NAME_BY_ID = {
    1: "Joint1",
    2: "Joint2",
    3: "Joint3",
    4: "Joint4",
    5: "Joint5",
    6: "Joint6",
}

PARAMS = [
    (0x7019, "pos"),
    (0x701A, "iqf"),
    (0x701B, "vel"),
    (0x701C, "vbus"),
]

DEFAULT_OFFSET_JSON = (
    "/home/zzj/sukinee_ws/vision_calibration/data/run_100_20260619_012357/"
    "gate5_result_77mm_exclude_P0021_P0008_joint2prior001/"
    "sukinee_motor_to_urdf_offsets_calibrated.json"
)

DEFAULT_OUT = "/tmp/sukinee_current_q.txt"
DEFAULT_JSON_OUT = "/tmp/sukinee_current_q_once.json"


def parse_joint_key_dict(raw: Dict[str, float]) -> Dict[int, float]:
    parsed: Dict[int, float] = {}

    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("Joint"):
            motor_id = int(key.replace("Joint", ""))
        else:
            motor_id = int(key)

        parsed[motor_id] = float(value)

    return parsed


def load_offset_json(path: Path) -> Tuple[Dict[int, float], Dict[int, float], Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "motor_to_urdf_sign" not in payload:
        raise RuntimeError("offset JSON missing key: motor_to_urdf_sign")
    if "motor_pos_at_urdf_zero" not in payload:
        raise RuntimeError("offset JSON missing key: motor_pos_at_urdf_zero")

    sign = parse_joint_key_dict(payload["motor_to_urdf_sign"])
    zero = parse_joint_key_dict(payload["motor_pos_at_urdf_zero"])

    for motor_id in ARM_JOINT_IDS:
        if motor_id not in sign:
            raise RuntimeError(f"offset JSON missing sign for Joint{motor_id}")
        if motor_id not in zero:
            raise RuntimeError(f"offset JSON missing zero for Joint{motor_id}")

    return sign, zero, payload


def validate_feedback(
    values: Dict[int, Dict[str, Optional[float]]],
    statuses: Dict[int, Dict[str, str]],
    motor_ids: List[int],
    min_vbus: float,
    max_vbus: float,
    max_abs_iqf: float,
    max_abs_vel: float,
) -> Tuple[bool, List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    for motor_id in motor_ids:
        for _index, name in PARAMS:
            status = statuses.get(motor_id, {}).get(name)
            value = values.get(motor_id, {}).get(name)

            if status != "OK":
                failures.append(f"Joint{motor_id} {name} read status={status}")
                continue

            if value is None:
                failures.append(f"Joint{motor_id} {name} value is None")

        if failures:
            continue

        vbus = float(values[motor_id]["vbus"])
        iqf = float(values[motor_id]["iqf"])
        vel = float(values[motor_id]["vel"])

        if not (min_vbus <= vbus <= max_vbus):
            failures.append(
                f"Joint{motor_id} vbus out of range: {vbus:.3f} V, "
                f"expected [{min_vbus:.3f}, {max_vbus:.3f}]"
            )

        if abs(iqf) > max_abs_iqf:
            failures.append(
                f"Joint{motor_id} iqf too large for current-q read: "
                f"{iqf:.3f} A > {max_abs_iqf:.3f} A"
            )

        if abs(vel) > max_abs_vel:
            failures.append(
                f"Joint{motor_id} velocity too large for stable current-q read: "
                f"{vel:.3f} rad/s > {max_abs_vel:.3f} rad/s"
            )

        if abs(vel) > 0.05:
            warnings.append(
                f"Joint{motor_id} is not fully still: vel={vel:+.6f} rad/s"
            )

    return len(failures) == 0, failures, warnings


def map_to_urdf_q(
    values: Dict[int, Dict[str, Optional[float]]],
    sign: Dict[int, float],
    zero: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    q_map: Dict[int, float] = {}
    qdot_map: Dict[int, float] = {}

    for motor_id in ARM_JOINT_IDS:
        motor_pos = float(values[motor_id]["pos"])
        motor_vel = float(values[motor_id]["vel"])

        q_map[motor_id] = sign[motor_id] * (motor_pos - zero[motor_id])
        qdot_map[motor_id] = sign[motor_id] * motor_vel

    return q_map, qdot_map


def format_current_q(q_map: Dict[int, float]) -> str:
    return ",".join(
        f"{JOINT_NAME_BY_ID[motor_id]}:{q_map[motor_id]:.9f}"
        for motor_id in ARM_JOINT_IDS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Sukinee Joint1~Joint6 current q_urdf once. Type17 read-only."
    )

    parser.add_argument("--can", default="can0", help="SocketCAN interface.")
    parser.add_argument(
        "--offset-json",
        default=DEFAULT_OFFSET_JSON,
        help="motor_pos -> q_urdf software offset JSON.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Text output path for COPY_THIS_CURRENT_Q string.",
    )
    parser.add_argument(
        "--json-out",
        default=DEFAULT_JSON_OUT,
        help="JSON output path for detailed read result.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.6,
        help="Per-parameter Type17 read timeout in seconds.",
    )
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=0.003,
        help="Delay between Type17 read requests.",
    )
    parser.add_argument(
        "--min-vbus",
        type=float,
        default=10.0,
        help="Minimum acceptable bus voltage.",
    )
    parser.add_argument(
        "--max-vbus",
        type=float,
        default=60.0,
        help="Maximum acceptable bus voltage.",
    )
    parser.add_argument(
        "--max-abs-iqf",
        type=float,
        default=1.0,
        help="Maximum acceptable absolute iqf for stable read.",
    )
    parser.add_argument(
        "--max-abs-vel",
        type=float,
        default=2.0,
        help="Maximum acceptable absolute motor velocity for stable read.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print raw motor feedback table.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    offset_json = Path(args.offset_json).expanduser().resolve()
    out_path = Path(args.out).expanduser()
    json_out_path = Path(args.json_out).expanduser()

    print("Sukinee read current-q once")
    print("  safety boundary:")
    print("    SocketCAN opened only for Type17 read-only parameter reads")
    print("    NO Type1 motion command")
    print("    NO Type3 enable")
    print("    NO Type4 disable")
    print("    NO Type6 zero setting")
    print("    NO Type18 parameter write")
    print("    NO save parameters")
    print("    NO CAN_ID change")
    print("    NO protocol switch")
    print("    NO MoveIt real execution")
    print()
    print(f"  can:         {args.can}")
    print(f"  offset_json: {offset_json}")
    print(f"  out:         {out_path}")
    print(f"  json_out:    {json_out_path}")
    print()

    sign, zero, offset_payload = load_offset_json(offset_json)

    driver = SukineeSocketCANDriver(args.can)

    values = {}
    statuses = {}
    started_at = time.time()

    try:
        driver.open()

        values, statuses = driver.read_many_params_float(
            motor_ids=ARM_JOINT_IDS,
            params=PARAMS,
            timeout=float(args.timeout),
            inter_request_delay=float(args.inter_request_delay),
        )

    except Exception as exc:
        print(f"RESULT: FAIL: {exc}")
        print("No Type1/Type3/Type4/Type6/Type18 command was sent by this script.")
        return 1

    finally:
        try:
            driver.close()
        except Exception:
            pass

    ok, failures, warnings = validate_feedback(
        values=values,
        statuses=statuses,
        motor_ids=ARM_JOINT_IDS,
        min_vbus=float(args.min_vbus),
        max_vbus=float(args.max_vbus),
        max_abs_iqf=float(args.max_abs_iqf),
        max_abs_vel=float(args.max_abs_vel),
    )

    if not ok:
        print("RESULT: FAIL")
        for item in failures:
            print("  -", item)

        summary = {
            "ok": False,
            "failures": failures,
            "warnings": warnings,
            "values": values,
            "statuses": statuses,
            "safety_boundary": {
                "socketcan_opened_for_type17_read": True,
                "type17_read_param_sent": True,
                "type1_sent": False,
                "type3_enable_sent": False,
                "type4_disable_sent": False,
                "type6_set_zero_sent": False,
                "type18_write_param_sent": False,
                "save_motor_parameters": False,
                "change_can_id": False,
                "switch_protocol": False,
                "moveit_real_execution": False,
            },
        }

        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Wrote JSON: {json_out_path}")
        print("No Type1/Type3/Type4/Type6/Type18 command was sent by this script.")
        return 2

    q_map, qdot_map = map_to_urdf_q(values, sign, zero)
    current_q = format_current_q(q_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(current_q + "\n", encoding="utf-8")

    finished_at = time.time()

    summary = {
        "ok": True,
        "current_q": current_q,
        "q_urdf_rad": {
            JOINT_NAME_BY_ID[mid]: q_map[mid] for mid in ARM_JOINT_IDS
        },
        "qdot_urdf_rad_s": {
            JOINT_NAME_BY_ID[mid]: qdot_map[mid] for mid in ARM_JOINT_IDS
        },
        "motor_feedback": {
            f"motor{mid}": {
                "pos_rad": values[mid]["pos"],
                "vel_rad_s": values[mid]["vel"],
                "iqf_A": values[mid]["iqf"],
                "vbus_V": values[mid]["vbus"],
                "statuses": statuses[mid],
                "sign": sign[mid],
                "motor_pos_at_urdf_zero": zero[mid],
            }
            for mid in ARM_JOINT_IDS
        },
        "warnings": warnings,
        "failures": [],
        "paths": {
            "offset_json": str(offset_json),
            "out": str(out_path),
            "json_out": str(json_out_path),
            "driver_dir": str(DRIVER_DIR),
        },
        "timing": {
            "started_unix": started_at,
            "finished_unix": finished_at,
            "elapsed_sec": finished_at - started_at,
        },
        "safety_boundary": {
            "socketcan_opened_for_type17_read": True,
            "type17_read_param_sent": True,
            "type1_sent": False,
            "type3_enable_sent": False,
            "type4_disable_sent": False,
            "type6_set_zero_sent": False,
            "type18_write_param_sent": False,
            "save_motor_parameters": False,
            "change_can_id": False,
            "switch_protocol": False,
            "moveit_real_execution": False,
        },
        "offset_payload_metadata": {
            k: v
            for k, v in offset_payload.items()
            if k not in ("motor_to_urdf_sign", "motor_pos_at_urdf_zero")
        },
    }

    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    json_out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("COPY_THIS_CURRENT_Q:")
    print(current_q)
    print()
    print(f"Wrote text: {out_path}")
    print(f"Wrote JSON: {json_out_path}")

    if args.print_raw:
        print()
        print("Raw feedback:")
        for mid in ARM_JOINT_IDS:
            print(
                f"  Joint{mid}: "
                f"motor_pos={values[mid]['pos']:+.9f} rad, "
                f"motor_vel={values[mid]['vel']:+.9f} rad/s, "
                f"iqf={values[mid]['iqf']:+.6f} A, "
                f"vbus={values[mid]['vbus']:+.3f} V, "
                f"q_urdf={q_map[mid]:+.9f} rad"
            )

    if warnings:
        print()
        print("Warnings:")
        for item in warnings:
            print("  -", item)

    print()
    print("RESULT: PASS")
    print("No Type1/Type3/Type4/Type6/Type18 command was sent by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

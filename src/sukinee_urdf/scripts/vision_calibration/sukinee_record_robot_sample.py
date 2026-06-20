#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_offsets(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    sign_map = obj.get("motor_to_urdf_sign")
    zero_map = obj.get("motor_pos_at_urdf_zero")

    if not isinstance(sign_map, dict):
        raise ValueError(
            f"offset file missing required dict field: motor_to_urdf_sign: {path}"
        )

    if not isinstance(zero_map, dict):
        raise ValueError(
            f"offset file missing required dict field: motor_pos_at_urdf_zero: {path}"
        )

    result = {}

    for name, zero in zero_map.items():
        if name not in sign_map:
            raise ValueError(
                f"offset file missing motor_to_urdf_sign for joint {name!r}: {path}"
            )

        result[str(name)] = {
            "sign": float(sign_map[name]),
            "motor_pos_at_urdf_zero": float(zero),
        }

    alias = {}
    for k, v in result.items():
        alias[k] = v
        alias[k.lower()] = v

    return alias, obj


class JointStateOnce(Node):
    def __init__(self, topic):
        super().__init__("sukinee_record_robot_sample")
        self.msg = None
        self.create_subscription(JointState, topic, self.cb, 10)

    def cb(self, msg):
        self.msg = msg


def select_joint_positions(msg: JointState, requested_names):
    pos_map = {}
    lower_to_real = {}

    for i, name in enumerate(msg.name):
        if i >= len(msg.position):
            continue
        pos_map[name] = float(msg.position[i])
        lower_to_real[name.lower()] = name

    selected = {}
    missing = []

    for name in requested_names:
        if name in pos_map:
            selected[name] = pos_map[name]
        elif name.lower() in lower_to_real:
            real_name = lower_to_real[name.lower()]
            selected[name] = pos_map[real_name]
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "joint_states 中缺少关节: "
            + ", ".join(missing)
            + "\n当前 /joint_states 可用关节名: "
            + ", ".join(msg.name)
        )

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", required=True, help="例如 S0007；必须和 vision sample 对齐")
    ap.add_argument("--topic", default="/joint_states")
    ap.add_argument("--timeout-sec", type=float, default=5.0)
    ap.add_argument("--joint-names", nargs="+", default=DEFAULT_JOINT_NAMES)
    ap.add_argument(
        "--offset-file",
        default="/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json",
    )
    ap.add_argument(
        "--out",
        default="/home/zzj/sukinee_ws/vision_calibration/data/robot_samples_trial.jsonl",
    )
    args = ap.parse_args()

    offset_path = Path(args.offset_file)
    if not offset_path.exists():
        raise FileNotFoundError(f"offset file not found: {offset_path}")

    offset_alias, offset_raw = load_offsets(offset_path)

    rclpy.init()
    node = JointStateOnce(args.topic)

    deadline = time.monotonic() + args.timeout_sec

    try:
        while node.msg is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise TimeoutError(f"timeout waiting for {args.topic}")

        msg = node.msg
    finally:
        node.destroy_node()
        rclpy.shutdown()

    q_urdf = select_joint_positions(msg, args.joint_names)

    motor_pos_est = {}
    offset_used = {}

    for joint_name, q in q_urdf.items():
        off = offset_alias.get(joint_name) or offset_alias.get(joint_name.lower())

        if off is None:
            motor_pos_est[joint_name] = None
            offset_used[joint_name] = None
            continue

        sign = float(off["sign"])
        zero = float(off["motor_pos_at_urdf_zero"])

        motor_pos = zero + sign * q

        motor_pos_est[joint_name] = motor_pos
        offset_used[joint_name] = {
            "sign": sign,
            "motor_pos_at_urdf_zero": zero,
        }

    record = {
        "protocol": "sukinee_robot_sample_v1",
        "sample_id": args.sample_id,
        "timestamp_robot_receive": now_iso(),
        "topic": args.topic,
        "joint_names": args.joint_names,
        "q_urdf_current_rad": q_urdf,
        "motor_pos_estimated_from_offset": motor_pos_est,
        "offset_file": str(offset_path),
        "offset_used": offset_used,
        "meaning": "robot joint state paired with vision sample by sample_id",
        "raw_joint_state": {
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        },
        "valid": True,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"[saved] {args.sample_id} -> {out_path}")
    print("[q_urdf_current_rad]")
    for k, v in q_urdf.items():
        print(f"  {k}: {v:.9f}")

    print("[motor_pos_estimated_from_offset]")
    for k, v in motor_pos_est.items():
        if v is None:
            print(f"  {k}: None")
        else:
            print(f"  {k}: {v:.9f}")


if __name__ == "__main__":
    main()

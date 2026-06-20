#!/usr/bin/env python3
"""
Inspect Sukinee URDF inertial parameters as loaded by Pinocchio.

Purpose:
  - Read the current URDF only.
  - Print Pinocchio joint/body names, parent-child relation, q/v indexes.
  - Print body inertial mass, COM lever, rotational inertia matrix.
  - Print subtree mass for each Joint1-Joint6 candidate.
  - Optionally write a JSON report for later inertia-correction design.

Safety:
  - This script does NOT use CAN.
  - This script does NOT read or command real motors.
  - This script does NOT publish ROS topics.
  - This script does NOT modify URDF or motor parameters.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_URDF = "/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"
DEFAULT_OUTPUT_JSON = "/home/zzj/sukinee_ws/sukinee_pinocchio_inertias_report.json"
ARM_JOINT_NAMES = [f"Joint{i}" for i in range(1, 7)]


def import_dependencies():
    try:
        import numpy as np
        import pinocchio as pin
    except Exception as exc:
        print("RESULT: FAIL")
        print("ERROR: failed to import pinocchio or numpy.")
        print("Try:")
        print('  python3 -c "import pinocchio as pin; import numpy as np; print(pin.__version__)"')
        raise exc
    return pin, np


def fmt_float(x: float) -> str:
    return f"{float(x):+.6f}"


def fmt_vec(values) -> str:
    return "[" + ", ".join(fmt_float(x) for x in values) + "]"


def to_float_list(values) -> List[float]:
    return [float(x) for x in list(values)]


def to_float_matrix(matrix) -> List[List[float]]:
    return [[float(x) for x in row] for row in matrix]


def frame_type_to_str(pin, frame_type: Any) -> str:
    # Works across Pinocchio versions where frame_type may be enum-like or int-like.
    try:
        name = frame_type.name
        if name:
            return str(name)
    except Exception:
        pass

    text = str(frame_type)
    if "." in text:
        return text.split(".")[-1]

    try:
        for name in ["OP_FRAME", "JOINT", "FIXED_JOINT", "BODY", "SENSOR"]:
            if hasattr(pin, "FrameType") and hasattr(pin.FrameType, name):
                if frame_type == getattr(pin.FrameType, name):
                    return name
    except Exception:
        pass

    return text


def collect_frames_by_parent_joint(pin, model) -> Dict[int, List[Dict[str, Any]]]:
    frames_by_joint: Dict[int, List[Dict[str, Any]]] = {jid: [] for jid in range(model.njoints)}

    for fid, frame in enumerate(model.frames):
        parent_joint = int(frame.parentJoint)
        parent_frame = int(frame.parentFrame)
        item = {
            "frame_id": int(fid),
            "name": str(frame.name),
            "type": frame_type_to_str(pin, frame.type),
            "parent_joint": parent_joint,
            "parent_frame": parent_frame,
        }
        frames_by_joint.setdefault(parent_joint, []).append(item)

    return frames_by_joint


def build_children(model) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {jid: [] for jid in range(model.njoints)}
    for jid in range(1, model.njoints):
        parent = int(model.parents[jid])
        children.setdefault(parent, []).append(jid)
    return children


def subtree_joint_ids(root_joint_id: int, children: Dict[int, List[int]]) -> List[int]:
    out: List[int] = []
    stack = [root_joint_id]
    while stack:
        jid = stack.pop()
        out.append(jid)
        stack.extend(reversed(children.get(jid, [])))
    return out


def joint_inertia_dict(np, model, jid: int) -> Dict[str, Any]:
    inertia = model.inertias[jid]
    mass = float(inertia.mass)
    lever = to_float_list(np.asarray(inertia.lever).reshape(-1))
    rotational_inertia = to_float_matrix(np.asarray(inertia.inertia))
    spatial_matrix = to_float_matrix(np.asarray(inertia.matrix()))

    return {
        "mass": mass,
        "com_lever_in_joint_frame_xyz_m": lever,
        "rotational_inertia_3x3": rotational_inertia,
        "spatial_inertia_6x6": spatial_matrix,
    }


def inspect_model(urdf_path: Path, show_all_frames: bool) -> Dict[str, Any]:
    pin, np = import_dependencies()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    children = build_children(model)
    frames_by_joint = collect_frames_by_parent_joint(pin, model)

    total_body_mass = 0.0
    for jid in range(1, model.njoints):
        total_body_mass += float(model.inertias[jid].mass)

    report: Dict[str, Any] = {
        "description": "Pinocchio inertial inspection report. Read-only; no CAN; no motor command.",
        "urdf_path": str(urdf_path),
        "model_summary": {
            "name": str(model.name),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "njoints": int(model.njoints),
            "nframes": int(model.nframes),
            "total_body_mass_excluding_universe_kg": total_body_mass,
        },
        "joints": [],
        "arm_joint_summary": [],
    }

    for jid in range(model.njoints):
        name = str(model.names[jid])
        parent = int(model.parents[jid]) if jid < len(model.parents) else -1
        child_ids = children.get(jid, [])
        inertia_info = joint_inertia_dict(np, model, jid)
        subtree_ids = subtree_joint_ids(jid, children) if jid != 0 else list(range(model.njoints))
        subtree_mass = sum(float(model.inertias[x].mass) for x in subtree_ids if x != 0)

        item = {
            "joint_id": int(jid),
            "joint_name": name,
            "parent_joint_id": parent,
            "parent_joint_name": str(model.names[parent]) if 0 <= parent < model.njoints else "",
            "child_joint_ids": [int(x) for x in child_ids],
            "child_joint_names": [str(model.names[x]) for x in child_ids],
            "nq": int(model.nqs[jid]),
            "nv": int(model.nvs[jid]),
            "idx_q": int(model.idx_qs[jid]),
            "idx_v": int(model.idx_vs[jid]),
            "inertia": inertia_info,
            "frames_attached_to_this_joint": frames_by_joint.get(jid, []),
            "subtree_joint_ids": [int(x) for x in subtree_ids],
            "subtree_joint_names": [str(model.names[x]) for x in subtree_ids],
            "subtree_mass_kg": float(subtree_mass),
        }
        report["joints"].append(item)

    for joint_name in ARM_JOINT_NAMES:
        if not model.existJointName(joint_name):
            report["arm_joint_summary"].append({
                "joint_name": joint_name,
                "exists": False,
                "warning": "Joint name not found in Pinocchio model.",
            })
            continue

        jid = int(model.getJointId(joint_name))
        joint_item = report["joints"][jid]
        frame_candidates = []
        for frame in joint_item["frames_attached_to_this_joint"]:
            ftype = frame["type"]
            fname = frame["name"]
            if fname != joint_name:
                frame_candidates.append(f"{fname}({ftype})")

        report["arm_joint_summary"].append({
            "joint_name": joint_name,
            "exists": True,
            "joint_id": jid,
            "parent_joint_name": joint_item["parent_joint_name"],
            "nq": joint_item["nq"],
            "nv": joint_item["nv"],
            "idx_q": joint_item["idx_q"],
            "idx_v": joint_item["idx_v"],
            "body_mass_kg": joint_item["inertia"]["mass"],
            "body_com_lever_in_joint_frame_xyz_m": joint_item["inertia"]["com_lever_in_joint_frame_xyz_m"],
            "subtree_mass_kg": joint_item["subtree_mass_kg"],
            "subtree_joint_names": joint_item["subtree_joint_names"],
            "attached_frame_candidates": frame_candidates,
            "recommended_correction_key": joint_name,
        })

    return report


def print_report(report: Dict[str, Any], show_all_frames: bool) -> None:
    summary = report["model_summary"]

    print()
    print("=" * 100)
    print("Sukinee Pinocchio inertial inspection")
    print("=" * 100)
    print(f"URDF path: {report['urdf_path']}")
    print(f"model name: {summary['name']}")
    print(
        f"nq={summary['nq']}, nv={summary['nv']}, "
        f"njoints={summary['njoints']}, nframes={summary['nframes']}"
    )
    print(f"total body mass excluding universe: {summary['total_body_mass_excluding_universe_kg']:.6f} kg")

    if summary["total_body_mass_excluding_universe_kg"] < 0.1:
        print("WARNING: total mass is very small. URDF inertial parameters may be missing or unrealistic.")

    print()
    print("-" * 100)
    print("Joint table")
    print("-" * 100)
    print("id | joint_name           | parent              | nq nv | idx_q idx_v | mass(kg) | subtree(kg)")
    print("---+----------------------+---------------------+-------+-------------+----------+------------")

    for item in report["joints"]:
        print(
            f"{item['joint_id']:>2} | "
            f"{item['joint_name']:<20} | "
            f"{item['parent_joint_name']:<19} | "
            f"{item['nq']:>1}  {item['nv']:>1} | "
            f"{item['idx_q']:>5} {item['idx_v']:>5} | "
            f"{item['inertia']['mass']:>8.5f} | "
            f"{item['subtree_mass_kg']:>10.5f}"
        )

    print()
    print("-" * 100)
    print("Arm Joint1-Joint6 inertial detail")
    print("-" * 100)

    for item in report["arm_joint_summary"]:
        name = item["joint_name"]
        if not item.get("exists", False):
            print(f"{name}: NOT FOUND")
            continue

        print(f"{name}:")
        print(f"  joint_id: {item['joint_id']}")
        print(f"  parent_joint: {item['parent_joint_name']}")
        print(f"  nq/nv: {item['nq']}/{item['nv']}, idx_q={item['idx_q']}, idx_v={item['idx_v']}")
        print(f"  body_mass: {item['body_mass_kg']:.6f} kg")
        print(f"  body_com_lever_in_joint_frame_xyz_m: {fmt_vec(item['body_com_lever_in_joint_frame_xyz_m'])}")
        print(f"  subtree_mass_from_this_joint: {item['subtree_mass_kg']:.6f} kg")
        print(f"  subtree_joints: {', '.join(item['subtree_joint_names'])}")
        if item["attached_frame_candidates"]:
            print(f"  attached frame candidates: {', '.join(item['attached_frame_candidates'])}")
        else:
            print("  attached frame candidates: none")
        print(f"  recommended correction key for first JSON: {item['recommended_correction_key']}")
        print()

    if show_all_frames:
        print("-" * 100)
        print("All frames grouped by parent joint")
        print("-" * 100)
        for joint in report["joints"]:
            frames = joint["frames_attached_to_this_joint"]
            if not frames:
                continue
            print(f"{joint['joint_name']}:")
            for frame in frames:
                print(
                    f"  frame_id={frame['frame_id']:>3}, "
                    f"name={frame['name']}, type={frame['type']}, "
                    f"parent_frame={frame['parent_frame']}"
                )

    print("=" * 100)
    print("Interpretation notes")
    print("=" * 100)
    print("1. This script only inspects the URDF as Pinocchio sees it. It does not use CAN or command motors.")
    print("2. model.inertias[joint_id] is the inertial body attached to that Pinocchio joint.")
    print("3. subtree_mass helps identify which upstream joints carry large downstream loads.")
    print("4. For the first inertia-correction JSON, using Joint2/Joint3/... as keys is safer than guessing link names.")
    print("5. If total mass or a major subtree mass is unrealistic, gravity compensation will be weak or biased.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Sukinee URDF inertial parameters loaded by Pinocchio. Read-only."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to static URDF file.")
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Where to save the JSON report. Use empty string to skip saving.",
    )
    parser.add_argument(
        "--show-all-frames",
        action="store_true",
        help="Print all Pinocchio frames grouped by parent joint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = Path(args.urdf).expanduser()

    print("Sukinee Pinocchio inertial inspection tool")
    print("Safety status:")
    print("  NO CAN access")
    print("  NO Type1 motion command")
    print("  NO Type3 enable")
    print("  NO Type4 disable")
    print("  NO Type17 read")
    print("  NO Type18 parameter write")
    print("  NO motor zero setting")
    print("  NO ROS topic publishing")
    print("  NO URDF modification")

    try:
        report = inspect_model(urdf_path, show_all_frames=args.show_all_frames)
        print_report(report, show_all_frames=args.show_all_frames)

        if args.output_json:
            output_path = Path(args.output_json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print()
            print(f"Saved JSON report: {output_path}")

        print()
        print("RESULT: PASS")
        return 0

    except Exception as exc:
        print()
        print("RESULT: FAIL")
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
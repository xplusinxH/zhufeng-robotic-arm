#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pinocchio as pin


ARM_JOINT_NAMES = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]

LOCK_JOINT_NAMES = [
    "Joint7",
    "left_finger",
    "right_finger",
]


def build_arm_model(urdf_path: Path):
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    q_full_neutral = pin.neutral(full_model)

    lock_joint_ids = []
    for name in LOCK_JOINT_NAMES:
        joint_id = full_model.getJointId(name)
        if joint_id >= len(full_model.joints):
            raise RuntimeError(f"Joint not found in URDF: {name}")
        lock_joint_ids.append(joint_id)

    arm_model = pin.buildReducedModel(full_model, lock_joint_ids, q_full_neutral)
    return arm_model


def make_q_from_arm_positions(model, positions):
    if len(positions) != 6:
        raise ValueError("Expected exactly 6 arm joint positions for Joint1-Joint6")

    q = pin.neutral(model)

    for joint_name, angle in zip(ARM_JOINT_NAMES, positions):
        joint_id = model.getJointId(joint_name)
        if joint_id >= len(model.joints):
            raise RuntimeError(f"Joint not found in reduced model: {joint_name}")

        joint = model.joints[joint_id]

        if joint.nq == 1:
            q[joint.idx_q] = angle
        elif joint.nq == 2 and joint.nv == 1:
            q[joint.idx_q] = np.cos(angle)
            q[joint.idx_q + 1] = np.sin(angle)
        else:
            raise RuntimeError(
                f"Unsupported joint representation for {joint_name}: "
                f"nq={joint.nq}, nv={joint.nv}"
            )

    return q


def main():
    parser = argparse.ArgumentParser(
        description="Compute arm-only gravity compensation torque for Sukinee Joint1-Joint6."
    )
    parser.add_argument(
        "--urdf",
        default="src/sukinee_urdf/urdf/sukinee_urdf.urdf",
        help="Path to URDF file.",
    )
    parser.add_argument(
        "--q",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="Joint1-Joint6 positions in radians.",
    )

    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    model = build_arm_model(urdf_path)
    data = model.createData()

    q = make_q_from_arm_positions(model, args.q)
    tau_g = pin.computeGeneralizedGravity(model, data, q)

    print("=== Sukinee arm-only gravity compensation ===")
    print(f"URDF: {urdf_path}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print()

    print("Input joint positions [rad]:")
    for name, value in zip(ARM_JOINT_NAMES, args.q):
        print(f"  {name}: {value: .6f}")
    print()

    print("Gravity compensation torque [Nm]:")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(f"  {name}: {tau_g[i]: .6f}")

    print()
    print("raw tau_g:", np.array2string(tau_g, precision=6, suppress_small=False))


if __name__ == "__main__":
    main()

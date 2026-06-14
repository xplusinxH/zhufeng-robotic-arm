#!/usr/bin/env python3
"""
Single-joint Type1 torque_ff sign probe for Sukinee.

Purpose:
  Verify whether +Type1 torque command increases or decreases q_urdf.

Why:
  motor_pos -> q_urdf feedback sign is NOT necessarily the same as
  tau_urdf -> motor torque command sign.

Safety:
  - Tests only ONE joint at a time.
  - Uses small torque pulses.
  - No MoveIt.
  - No ros2_control real hardware interface.
  - No zero setting.
  - No save parameters.
  - No CAN_ID change.
  - Always sends Type4 disable at the end if armed commands were sent.

This is NOT a zero-torque drag run.
Do not use a 6-second duration here. This is a short sign probe.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

from sukinee_socketcan_driver import SukineeSocketCANDriver, CAN_IFACE_DEFAULT


POS_INDEX = 0x7019
IQF_INDEX = 0x701A
VEL_INDEX = 0x701B
VBUS_INDEX = 0x701C

ARM_JOINT_IDS = [1, 2, 3, 4, 5, 6]
TESTABLE_JOINTS = [2, 3, 4, 5, 6]

DEFAULT_OFFSET_JSON = "/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json"

CONFIRM_TEXT = "I_UNDERSTAND_THIS_PROBES_SINGLE_JOINT_TORQUE_SIGN"


def parse_joint_key_dict(raw: Dict[str, float]) -> Dict[int, float]:
    parsed: Dict[int, float] = {}

    for key, value in raw.items():
        if isinstance(key, str) and key.startswith("Joint"):
            motor_id = int(key.replace("Joint", ""))
        else:
            motor_id = int(key)

        parsed[motor_id] = float(value)

    return parsed


def load_offset_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "motor_to_urdf_sign" not in payload:
        raise RuntimeError("offset JSON missing key: motor_to_urdf_sign")
    if "motor_pos_at_urdf_zero" not in payload:
        raise RuntimeError("offset JSON missing key: motor_pos_at_urdf_zero")

    position_sign = parse_joint_key_dict(payload["motor_to_urdf_sign"])
    zero = parse_joint_key_dict(payload["motor_pos_at_urdf_zero"])

    for motor_id in ARM_JOINT_IDS:
        if motor_id not in position_sign:
            raise RuntimeError(f"offset JSON missing sign for Joint{motor_id}")
        if motor_id not in zero:
            raise RuntimeError(f"offset JSON missing zero for Joint{motor_id}")

    return position_sign, zero, payload


def motor_pos_to_q_urdf(motor_id: int, motor_pos: float, position_sign, zero) -> float:
    return float(position_sign[motor_id]) * (float(motor_pos) - float(zero[motor_id]))


def read_param(driver: SukineeSocketCANDriver, motor_id: int, index: int, timeout: float) -> float:
    status, value = driver.read_param_float(motor_id=motor_id, index=index, timeout=timeout)
    if status != "OK":
        raise RuntimeError(f"Joint{motor_id} read index 0x{index:04X} failed: {status}")
    return float(value)


def read_joint_feedback(driver: SukineeSocketCANDriver, motor_id: int, timeout: float):
    pos = read_param(driver, motor_id, POS_INDEX, timeout)
    iqf = read_param(driver, motor_id, IQF_INDEX, timeout)
    vel = read_param(driver, motor_id, VEL_INDEX, timeout)
    vbus = read_param(driver, motor_id, VBUS_INDEX, timeout)
    return {
        "pos": pos,
        "iqf": iqf,
        "vel": vel,
        "vbus": vbus,
    }


def feedback_sanity_check(motor_id: int, fb) -> Tuple[bool, str]:
    vbus = float(fb["vbus"])
    iqf = float(fb["iqf"])
    vel = float(fb["vel"])

    if not (40.0 <= vbus <= 55.0):
        return False, f"Joint{motor_id} vbus out of range: {vbus:.3f} V"
    if abs(iqf) > 0.5:
        return False, f"Joint{motor_id} iqf too large before probe: {iqf:.3f} A"
    if abs(vel) > 0.8:
        return False, f"Joint{motor_id} velocity too large before probe: {vel:.3f} rad/s"

    return True, "OK"


def send_zero_torque_for(
    driver: SukineeSocketCANDriver,
    motor_id: int,
    hold_pos: float,
    seconds: float,
    rate: float,
):
    period = 1.0 / rate
    deadline = time.monotonic() + max(0.0, seconds)

    while time.monotonic() < deadline:
        driver.send_motion_control(
            motor_id=motor_id,
            position=float(hold_pos),
            velocity=0.0,
            kp=0.0,
            kd=0.0,
            torque=0.0,
        )
        time.sleep(period)


def send_torque_pulse(
    driver: SukineeSocketCANDriver,
    motor_id: int,
    hold_pos: float,
    torque: float,
    pulse_sec: float,
    rate: float,
):
    period = 1.0 / rate
    deadline = time.monotonic() + max(0.0, pulse_sec)

    count = 0
    while time.monotonic() < deadline:
        driver.send_motion_control(
            motor_id=motor_id,
            position=float(hold_pos),
            velocity=0.0,
            kp=0.0,
            kd=0.0,
            torque=float(torque),
        )
        count += 1
        time.sleep(period)

    return count


def classify_sign(delta_q_plus: float, delta_q_minus: float, threshold: float):
    """
    We compare +pulse displacement against -pulse displacement.

    score = delta_q_plus - delta_q_minus

    If score > threshold:
        +motor torque tends to increase q_urdf.
        torque_output_sign should be +1.0 for tau_motor = sign * tau_urdf.

    If score < -threshold:
        +motor torque tends to decrease q_urdf.
        torque_output_sign should be -1.0.

    Otherwise:
        result uncertain.
    """
    score = float(delta_q_plus) - float(delta_q_minus)

    if score > threshold:
        return "POSITIVE_TORQUE_INCREASES_Q", +1.0, score
    if score < -threshold:
        return "POSITIVE_TORQUE_DECREASES_Q", -1.0, score

    return "UNCERTAIN_SMALL_MOTION", 0.0, score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-joint Type1 torque_ff command sign probe."
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument("--offset-json", default=DEFAULT_OFFSET_JSON)

    parser.add_argument(
        "--joint",
        type=int,
        required=True,
        help="Joint to test. Recommended: 2, 3, or 4 first. Joint7 is not allowed.",
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=0.12,
        help="Abs torque pulse in Nm. Recommended: J2/J3 0.12, J4 0.05~0.08.",
    )
    parser.add_argument(
        "--pulse-sec",
        type=float,
        default=0.20,
        help="Pulse duration in seconds. This is a short sign probe, not a 6-second gravity run.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.20,
        help="Zero torque settle time between pulses.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=80.0,
        help="Type1 command send rate during pulse.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.10,
        help="Type17 read timeout.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Minimum q_urdf delta score in rad to classify sign.",
    )
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm", default="")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    motor_id = int(args.joint)
    offset_path = Path(args.offset_json).expanduser()

    print("Sukinee single-joint Type1 torque sign probe")
    print()
    print("Safety status:")
    print("  This script can send REAL Type1 torque_ff pulses when armed.")
    print("  Tests only ONE joint.")
    print("  NO MoveIt real execution.")
    print("  NO ros2_control real hardware interface.")
    print("  NO zero setting.")
    print("  NO save parameters.")
    print("  NO CAN_ID change.")
    print("  finally path sends Type4 disable to the tested joint.")
    print()

    if motor_id not in TESTABLE_JOINTS:
        print("RESULT: FAIL")
        print(f"ERROR: --joint must be one of {TESTABLE_JOINTS}. Joint7 is not allowed.")
        return 2

    if not offset_path.exists():
        print("RESULT: FAIL")
        print(f"ERROR: offset JSON not found: {offset_path}")
        return 2

    if args.torque <= 0.0 or args.torque > 2:
        print("RESULT: FAIL")
        print("ERROR: --torque must be >0 and <=2 Nm for this sign probe.")
        return 2

    if args.pulse_sec <= 0.0 or args.pulse_sec > 3:
        print("RESULT: FAIL")
        print("ERROR: --pulse-sec must be >0 and <=3 seconds.")
        return 2

    if args.settle_sec < 0.0 or args.settle_sec > 4:
        print("RESULT: FAIL")
        print("ERROR: --settle-sec must be between 0 and 4 second.")
        return 2

    if args.rate <= 10.0 or args.rate > 150.0:
        print("RESULT: FAIL")
        print("ERROR: --rate must be >10 and <=150 Hz.")
        return 2

    position_sign, zero, _payload = load_offset_json(offset_path)

    print("Loaded offset JSON:")
    print(f"  {offset_path}")
    print(f"  Joint{motor_id}: position_sign={position_sign[motor_id]:+.1f}, software_zero={zero[motor_id]:+.9f}")
    print()

    print("Probe settings:")
    print(f"  joint:       Joint{motor_id}")
    print(f"  torque:      ±{args.torque:.4f} Nm")
    print(f"  pulse_sec:   {args.pulse_sec:.3f} s")
    print(f"  settle_sec:  {args.settle_sec:.3f} s")
    print(f"  rate:        {args.rate:.1f} Hz")
    print(f"  threshold:   {args.threshold:.6f} rad")
    print()

    driver = SukineeSocketCANDriver(args.can)
    did_send_real_command = False

    try:
        driver.open()

        print("=" * 90)
        print("Pre-check: Type17 read-only feedback")
        print("=" * 90)

        fb0 = read_joint_feedback(driver, motor_id, args.timeout)
        ok, reason = feedback_sanity_check(motor_id, fb0)
        if not ok:
            print("RESULT: FAIL")
            print(f"Feedback sanity check failed: {reason}")
            return 1

        pos0 = fb0["pos"]
        q0 = motor_pos_to_q_urdf(motor_id, pos0, position_sign, zero)

        print(f"Joint{motor_id} feedback before probe:")
        print(f"  motor_pos: {pos0:+.9f} rad")
        print(f"  q_urdf:    {q0:+.9f} rad")
        print(f"  iqf:       {fb0['iqf']:+.6f} A")
        print(f"  vel:       {fb0['vel']:+.6f} rad/s")
        print(f"  vbus:      {fb0['vbus']:+.6f} V")
        print()

        if not args.armed:
            print("DRY-RUN ONLY. No Type3 / Type1 / Type4 command was sent.")
            print("To actually run this single-joint sign probe:")
            print(f"  --armed --confirm {CONFIRM_TEXT}")
            print()
            print("RESULT: DRY_RUN_PASS")
            return 0

        if args.confirm != CONFIRM_TEXT:
            print("RESULT: FAIL")
            print("ERROR: armed mode requires exact confirmation:")
            print(f"  --confirm {CONFIRM_TEXT}")
            return 2

        did_send_real_command = True

        print("=" * 90)
        print("ARMED: single-joint torque sign probe")
        print("=" * 90)
        print("Keep one hand on the arm. Press Ctrl+C immediately if direction feels wrong.")
        print("This is a short pulse probe, not a normal zero-torque drag run.")
        print()

        # Prepare only the tested joint.
        driver.send_disable(motor_id, clear_fault=False)
        time.sleep(0.05)

        driver.send_set_motion_mode(motor_id)
        time.sleep(0.05)

        driver.send_enable(motor_id)
        time.sleep(0.05)

        # First pulse: positive torque.
        fb_before_plus = read_joint_feedback(driver, motor_id, args.timeout)
        pos_before_plus = fb_before_plus["pos"]
        q_before_plus = motor_pos_to_q_urdf(motor_id, pos_before_plus, position_sign, zero)

        print(f"Positive pulse: +{args.torque:.4f} Nm")
        count_plus = send_torque_pulse(
            driver=driver,
            motor_id=motor_id,
            hold_pos=pos_before_plus,
            torque=+args.torque,
            pulse_sec=args.pulse_sec,
            rate=args.rate,
        )
        send_zero_torque_for(driver, motor_id, pos_before_plus, args.settle_sec, args.rate)

        fb_after_plus = read_joint_feedback(driver, motor_id, args.timeout)
        pos_after_plus = fb_after_plus["pos"]
        q_after_plus = motor_pos_to_q_urdf(motor_id, pos_after_plus, position_sign, zero)

        delta_motor_plus = pos_after_plus - pos_before_plus
        delta_q_plus = q_after_plus - q_before_plus

        print(f"  frames_sent:       {count_plus}")
        print(f"  motor_pos_before:  {pos_before_plus:+.9f}")
        print(f"  motor_pos_after:   {pos_after_plus:+.9f}")
        print(f"  delta_motor_pos:   {delta_motor_plus:+.9f} rad")
        print(f"  q_before:          {q_before_plus:+.9f}")
        print(f"  q_after:           {q_after_plus:+.9f}")
        print(f"  delta_q_urdf:      {delta_q_plus:+.9f} rad")
        print()

        # Second pulse: negative torque.
        fb_before_minus = read_joint_feedback(driver, motor_id, args.timeout)
        pos_before_minus = fb_before_minus["pos"]
        q_before_minus = motor_pos_to_q_urdf(motor_id, pos_before_minus, position_sign, zero)

        print(f"Negative pulse: -{args.torque:.4f} Nm")
        count_minus = send_torque_pulse(
            driver=driver,
            motor_id=motor_id,
            hold_pos=pos_before_minus,
            torque=-args.torque,
            pulse_sec=args.pulse_sec,
            rate=args.rate,
        )
        send_zero_torque_for(driver, motor_id, pos_before_minus, args.settle_sec, args.rate)

        fb_after_minus = read_joint_feedback(driver, motor_id, args.timeout)
        pos_after_minus = fb_after_minus["pos"]
        q_after_minus = motor_pos_to_q_urdf(motor_id, pos_after_minus, position_sign, zero)

        delta_motor_minus = pos_after_minus - pos_before_minus
        delta_q_minus = q_after_minus - q_before_minus

        print(f"  frames_sent:       {count_minus}")
        print(f"  motor_pos_before:  {pos_before_minus:+.9f}")
        print(f"  motor_pos_after:   {pos_after_minus:+.9f}")
        print(f"  delta_motor_pos:   {delta_motor_minus:+.9f} rad")
        print(f"  q_before:          {q_before_minus:+.9f}")
        print(f"  q_after:           {q_after_minus:+.9f}")
        print(f"  delta_q_urdf:      {delta_q_minus:+.9f} rad")
        print()

        driver.send_disable(motor_id, clear_fault=False)
        time.sleep(0.05)

        result, torque_output_sign, score = classify_sign(
            delta_q_plus=delta_q_plus,
            delta_q_minus=delta_q_minus,
            threshold=args.threshold,
        )

        print("=" * 90)
        print("Probe result")
        print("=" * 90)
        print(f"Joint{motor_id}:")
        print(f"  delta_q_plus:        {delta_q_plus:+.9f} rad")
        print(f"  delta_q_minus:       {delta_q_minus:+.9f} rad")
        print(f"  score plus-minus:    {score:+.9f} rad")
        print(f"  classification:      {result}")

        if torque_output_sign == 0.0:
            print("  torque_output_sign:  UNCERTAIN")
            print()
            print("RESULT: WARN")
            print("Motion was too small or ambiguous. Try slightly larger --torque or --pulse-sec, still one joint only.")
        else:
            print(f"  suggested torque_output_sign Joint{motor_id}: {torque_output_sign:+.1f}")
            print()
            print("Meaning:")
            if torque_output_sign > 0:
                print("  +Type1 torque tends to increase q_urdf.")
                print("  Use tau_motor = +1.0 * tau_urdf for this joint.")
            else:
                print("  +Type1 torque tends to decrease q_urdf.")
                print("  Use tau_motor = -1.0 * tau_urdf for this joint.")
            print()
            print("RESULT: PASS")

        return 0

    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt: disabling tested joint if needed.")
        if did_send_real_command:
            try:
                driver.send_disable(motor_id, clear_fault=False)
            except Exception as e:
                print(f"Disable after KeyboardInterrupt failed: {e}")
        print("RESULT: INTERRUPTED")
        return 130

    except Exception as exc:
        print()
        print(f"ERROR: {exc}")
        if did_send_real_command:
            print("Exception path: sending Type4 disable to tested joint.")
            try:
                driver.send_disable(motor_id, clear_fault=False)
            except Exception as e:
                print(f"Disable after exception failed: {e}")
        print("RESULT: FAIL")
        return 1

    finally:
        if did_send_real_command:
            try:
                driver.send_disable(motor_id, clear_fault=False)
            except Exception:
                pass
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
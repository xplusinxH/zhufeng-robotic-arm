#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time


CAN_IFACE_DEFAULT = "can0"
HOST_ID = 0xFD
COMM_DISABLE = 4

CONFIRM_TEXT = "STOP_ALL_MOTORS_NOW"

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]


def build_extended_can_id(comm_type: int, data_area2: int, target_id: int) -> int:
    return ((comm_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (target_id & 0xFF)


def make_disable_frame(motor_id: int, clear_fault: bool = False) -> str:
    can_id = build_extended_can_id(COMM_DISABLE, HOST_ID, motor_id)
    first_byte = 1 if clear_fault else 0
    data = f"{first_byte:02X}" + "00" * 7
    return f"{can_id:08X}#{data}"


def send_frame(can_iface: str, frame: str) -> None:
    subprocess.run(
        ["cansend", can_iface, frame],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send Type4 disable/stop frames to Sukinee motors."
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--clear-fault",
        action="store_true",
        help="Set byte0=1 in Type4 disable frame. Default is false.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="How many stop sweeps to send. Default: 2.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.03,
        help="Delay between frames in seconds. Default: 0.03.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Sukinee stop-all motors tool")
    print()
    print("This tool sends only Type4 disable/stop frames.")
    print("NO enable")
    print("NO Type1 motion command")
    print("NO Type18 parameter write")
    print("NO zero setting")
    print("NO MoveIt real execution")
    print()

    frames = [
        (mid, make_disable_frame(mid, clear_fault=args.clear_fault))
        for mid in MOTOR_IDS
    ]

    print("Frames:")
    for mid, frame in frames:
        print(f"  Joint{mid}: {frame}")

    if not args.armed:
        print()
        print("DRY-RUN ONLY. No CAN frame was sent.")
        print("To actually stop all motors, run with:")
        print(f"  --armed --confirm {CONFIRM_TEXT}")
        return 0

    if args.confirm != CONFIRM_TEXT:
        print()
        print("ERROR: armed mode requires exact confirmation:")
        print(f"  --confirm {CONFIRM_TEXT}")
        return 2

    if args.repeat < 1 or args.repeat > 10:
        print("ERROR: --repeat must be between 1 and 10.")
        return 2

    print()
    print("ARMED: sending Type4 disable/stop frames to Joint1-Joint7.")
    print("Press Ctrl+C now to abort.")
    for sec in range(3, 0, -1):
        print(f"  sending in {sec}...")
        time.sleep(1.0)

    try:
        for sweep in range(args.repeat):
            print(f"Stop sweep {sweep + 1}/{args.repeat}")
            for mid, frame in frames:
                print(f"  SENDING Joint{mid}: {frame}")
                send_frame(args.can, frame)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        print("ABORTED BY USER")
        return 130
    except Exception as e:
        print()
        print(f"STOP-ALL RESULT: FAIL: {e}")
        return 1

    print()
    print("STOP-ALL RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
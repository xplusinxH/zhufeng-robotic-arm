#!/usr/bin/env python3
import re
import select
import struct
import subprocess
import time
from collections import defaultdict


CAN_IFACE = "can0"

# Temporary for the current hardware situation:
# Joint5 terminal is known faulty. Set this to set() after Joint5 is repaired.
SKIP_MOTOR_IDS = set()

READ_TIMEOUT_SEC = 2
MAX_READ_ATTEMPTS = 2
RETRY_DELAY_SEC = 0.10
INTER_PARAM_DELAY_SEC = 0.08
INTER_MOTOR_DELAY_SEC = 0.10
LOOP_DELAY_SEC = 1.0

# Joint name, motor type, CAN_ID
MOTORS = [
    ("Joint1", "RS00", 0x01),
    ("Joint2", "RS00", 0x02),
    ("Joint3", "RS00", 0x03),
    ("Joint4", "RS05", 0x04),
    ("Joint5", "RS05", 0x05),
    ("Joint6", "RS05", 0x06),
    ("Joint7", "RS05", 0x07),
]

# index, name, unit
PARAMS = [
    (0x7019, "pos", "rad"),
    (0x701A, "iqf", "A"),
    (0x701B, "vel", "rad/s"),
    (0x701C, "vbus", "V"),
]


def make_read_frame(motor_id: int, index: int) -> str:
    # Communication type 17:
    # CAN ID: 0x11 00 FD motor_id
    # DATA: index little-endian + 00 00 + 00 00 00 00
    can_id = f"1100FD{motor_id:02X}"
    index_lo = index & 0xFF
    index_hi = (index >> 8) & 0xFF
    data = f"{index_lo:02X}{index_hi:02X}000000000000"
    return f"{can_id}#{data}"


def send_read(motor_id: int, index: int) -> None:
    frame = make_read_frame(motor_id, index)
    subprocess.run(
        ["cansend", CAN_IFACE, frame],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_float_from_parts(parts) -> float:
    if len(parts) != 8:
        raise ValueError(f"Expected 8 bytes, got: {' '.join(parts)}")

    raw = bytes(int(x, 16) for x in parts[4:8])
    return struct.unpack("<f", raw)[0]


def parse_candump_line(line: str):
    if "[8]" not in line:
        return None

    left, data_hex = line.split("[8]", 1)
    parts = data_hex.strip().split()

    if len(parts) != 8:
        return None

    can_id = None
    for token in reversed(left.strip().split()):
        token = token.strip().upper()
        if re.fullmatch(r"[0-9A-F]{3,8}", token):
            can_id = token
            break

    if can_id is None:
        return None

    return can_id, parts


class CandumpReader:
    def __init__(self):
        # Use one long-running candump process instead of starting one process per parameter.
        # We intentionally listen to the whole bus, then filter in Python.
        self.proc = subprocess.Popen(
            ["candump", "-td", CAN_IFACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        time.sleep(0.10)

        if self.proc.poll() is not None:
            _, err = self.proc.communicate()
            raise RuntimeError(f"candump exited early: {err.strip()}")

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=1.0)

    def drain_available(self):
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("candump process exited unexpectedly.")

            ready, _, _ = select.select([self.proc.stdout], [], [], 0.0)
            if not ready:
                return

            line = self.proc.stdout.readline()
            if not line:
                return

    def readline_until(self, deadline: float):
        if self.proc.poll() is not None:
            raise RuntimeError("candump process exited unexpectedly.")

        timeout = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)

        if not ready:
            return None

        return self.proc.stdout.readline()


def read_param(reader: CandumpReader, motor_id: int, index: int):
    success_resp_id = f"1100{motor_id:02X}FD"
    fail_resp_id = f"1101{motor_id:02X}FD"
    expected_index = f"{index & 0xFF:02X}{(index >> 8) & 0xFF:02X}"

    for attempt in range(1, MAX_READ_ATTEMPTS + 1):
        # Remove old frames before each request so a late reply from a previous
        # timeout does not get mistaken as the current reply.
        reader.drain_available()

        try:
            send_read(motor_id, index)
        except subprocess.CalledProcessError:
            return "SEND_ERROR", None

        deadline = time.monotonic() + READ_TIMEOUT_SEC

        while time.monotonic() < deadline:
            line = reader.readline_until(deadline)
            if line is None:
                break

            parsed = parse_candump_line(line)
            if parsed is None:
                continue

            can_id, parts = parsed

            if can_id not in (success_resp_id, fail_resp_id):
                continue

            rx_index = (parts[0] + parts[1]).upper()
            if rx_index != expected_index:
                continue

            if can_id == fail_resp_id:
                return "READ_FAIL", None

            try:
                value = parse_float_from_parts(parts)
            except Exception:
                return "PARSE_ERROR", None

            return "OK", value

        if attempt < MAX_READ_ATTEMPTS:
            time.sleep(RETRY_DELAY_SEC)

    return "TIMEOUT", None


def format_value(status: str, value, unit: str) -> str:
    if status == "OK":
        return f"{value:.6f} {unit}"
    return status


def read_motor(reader: CandumpReader, motor_id: int):
    values = {}
    statuses = {}

    if motor_id in SKIP_MOTOR_IDS:
        for _, name, _ in PARAMS:
            values[name] = "SKIPPED_OFFLINE"
            statuses[name] = "SKIPPED_OFFLINE"
        return values, statuses

    for index, name, unit in PARAMS:
        status, value = read_param(reader, motor_id, index)
        statuses[name] = status
        values[name] = format_value(status, value, unit)
        time.sleep(INTER_PARAM_DELAY_SEC)

    return values, statuses


def print_stats(stats):
    total_ok = sum(v["OK"] for v in stats.values())
    total_timeout = sum(v["TIMEOUT"] for v in stats.values())
    total_read_fail = sum(v["READ_FAIL"] for v in stats.values())
    total_send_error = sum(v["SEND_ERROR"] for v in stats.values())
    total_parse_error = sum(v["PARSE_ERROR"] for v in stats.values())

    print(
        "stats | "
        f"OK={total_ok} "
        f"TIMEOUT={total_timeout} "
        f"READ_FAIL={total_read_fail} "
        f"SEND_ERROR={total_send_error} "
        f"PARSE_ERROR={total_parse_error}"
    )


def main():
    print("Joint1-Joint7 read-only monitor")
    print("NO enable, NO torque command, NO current command, NO zero setting, NO motion command")
    print("Reading feedback only: pos / iqf / vel / vbus")
    print(f"CAN interface: {CAN_IFACE}")
    print(f"Read timeout: {READ_TIMEOUT_SEC:.2f} s, attempts: {MAX_READ_ATTEMPTS}")
    print(f"Skipped motor IDs: {sorted(SKIP_MOTOR_IDS)}")
    print("Press Ctrl+C to stop")
    print()

    stats = defaultdict(lambda: defaultdict(int))
    cycle = 0
    reader = CandumpReader()

    try:
        while True:
            cycle += 1
            print("=" * 100)
            print(f"cycle={cycle}")

            for joint_name, motor_type, motor_id in MOTORS:
                values, statuses = read_motor(reader, motor_id)

                for _, name, _ in PARAMS:
                    status = statuses[name]
                    if status in ("OK", "TIMEOUT", "READ_FAIL", "SEND_ERROR", "PARSE_ERROR"):
                        stats[(motor_id, name)][status] += 1

                print(
                    f"{joint_name:<6} {motor_type:<4} CAN_ID={motor_id:<2} | "
                    f"pos={values['pos']:<18} | "
                    f"iqf={values['iqf']:<18} | "
                    f"vel={values['vel']:<18} | "
                    f"vbus={values['vbus']}"
                )

                time.sleep(INTER_MOTOR_DELAY_SEC)

            print_stats(stats)
            time.sleep(LOOP_DELAY_SEC)

    finally:
        reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. No enable, no torque/current command, no zero setting, no motion command was sent.")
#!/usr/bin/env python3
import argparse
import statistics
import sys
import time
from typing import Dict, List, Tuple

from sukinee_socketcan_driver import SukineeSocketCANDriver, CAN_IFACE_DEFAULT


PARAM_INDEX_BY_NAME = {
    "pos": 0x7019,
    "iqf": 0x701A,
    "vel": 0x701B,
    "vbus": 0x701C,
}


def parse_int_list(text: str) -> List[int]:
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            a = int(a)
            b = int(b)
            if b < a:
                raise ValueError(f"invalid range: {item}")
            result.extend(range(a, b + 1))
        else:
            result.append(int(item))
    return result


def parse_param_list(text: str) -> List[Tuple[int, str]]:
    text = text.strip().lower()

    if text == "pos":
        names = ["pos"]
    elif text == "all":
        names = ["pos", "iqf", "vel", "vbus"]
    else:
        names = [x.strip().lower() for x in text.split(",") if x.strip()]

    params = []
    for name in names:
        if name not in PARAM_INDEX_BY_NAME:
            raise ValueError(
                f"unknown param '{name}', supported: pos, iqf, vel, vbus, all"
            )
        params.append((PARAM_INDEX_BY_NAME[name], name))

    return params


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * p / 100.0))
    idx = max(0, min(len(values_sorted) - 1, idx))
    return values_sorted[idx]


def format_value(value):
    if value is None:
        return "None"
    return f"{value:+.6f}"


def run_once(driver, motors, params, timeout, inter_request_delay):
    values: Dict[int, Dict[str, float]] = {}
    statuses: Dict[int, Dict[str, str]] = {}

    ok_count = 0
    fail_count = 0
    timeout_count = 0

    for motor_id in motors:
        values[motor_id] = {}
        statuses[motor_id] = {}

        for index, name in params:
            status, value = driver.read_param_float(
                motor_id=motor_id,
                index=index,
                timeout=timeout,
            )

            statuses[motor_id][name] = status
            values[motor_id][name] = value

            if status == "OK":
                ok_count += 1
            else:
                fail_count += 1
                if status == "TIMEOUT":
                    timeout_count += 1

            if inter_request_delay > 0:
                time.sleep(inter_request_delay)

    return values, statuses, ok_count, fail_count, timeout_count


def print_last_values(values, statuses, motors, params):
    print("Last values:")
    for motor_id in motors:
        items = []
        for _index, name in params:
            st = statuses[motor_id].get(name, "MISSING")
            value = values[motor_id].get(name)
            if st == "OK":
                items.append(f"{name}={format_value(value)}")
            else:
                items.append(f"{name}={st}")
        print(f"  Joint{motor_id}: " + " | ".join(items))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Native SocketCAN Type17 read-rate test. "
            "Read-only: no Type1, no Type3, no Type4, no Type18."
        )
    )
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument(
        "--motors",
        default="1-6",
        help="Comma/range list, e.g. 1-6, 1-7, or 2,3,4. Default: 1-6.",
    )
    parser.add_argument(
        "--params",
        default="pos",
        help="pos, all, or comma list: pos,iqf,vel,vbus. Default: pos.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Test duration in seconds.",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=0.0,
        help=(
            "Paced loop target cycle rate in Hz. "
            "0 means run as fast as possible. "
            "One cycle reads all selected motors and params."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.25,
        help="Timeout for each Type17 parameter read, seconds.",
    )
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=0.0,
        help="Delay after each Type17 request, seconds. Default: 0.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print progress every N cycles.",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=3,
        help="Warmup cycles before statistics. Default: 3.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print last read values at the end.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        motors = parse_int_list(args.motors)
        params = parse_param_list(args.params)
    except Exception as e:
        print(f"RESULT: FAIL")
        print(f"Argument parse error: {e}")
        return 2

    for motor_id in motors:
        if motor_id < 1 or motor_id > 7:
            print("RESULT: FAIL")
            print(f"Invalid motor id: {motor_id}. Expected 1..7.")
            return 2

    if args.seconds <= 0:
        print("RESULT: FAIL")
        print("--seconds must be > 0.")
        return 2

    if args.target_rate < 0:
        print("RESULT: FAIL")
        print("--target-rate must be >= 0.")
        return 2

    if args.timeout <= 0:
        print("RESULT: FAIL")
        print("--timeout must be > 0.")
        return 2

    request_count_per_cycle = len(motors) * len(params)

    print("Sukinee SocketCAN Type17 read-rate test")
    print()
    print("Safety status:")
    print("  NO Type1 motion command")
    print("  NO Type3 enable")
    print("  NO Type4 disable")
    print("  NO Type18 parameter write")
    print("  NO Type6 zero setting")
    print("  Native SocketCAN only; no cansend/candump subprocess.")
    print()
    print(f"CAN interface: {args.can}")
    print(f"Motors: {motors}")
    print(f"Params: {[name for _idx, name in params]}")
    print(f"Requests per cycle: {request_count_per_cycle}")
    print(f"Duration: {args.seconds:.3f} s")
    if args.target_rate > 0:
        print(f"Mode: paced target {args.target_rate:.2f} cycle/s")
    else:
        print("Mode: maximum throughput")
    print(f"Timeout per request: {args.timeout:.3f} s")
    print(f"Inter-request delay: {args.inter_request_delay:.6f} s")
    print()

    driver = SukineeSocketCANDriver(args.can)

    cycle_times = []
    ok_counts = []
    fail_counts = []
    timeout_counts = []
    overrun_count = 0

    total_ok = 0
    total_fail = 0
    total_timeout = 0
    cycle = 0

    last_values = None
    last_statuses = None

    try:
        driver.open()

        print(f"Warmup cycles: {args.warmup_cycles}")
        for _ in range(args.warmup_cycles):
            run_once(
                driver,
                motors=motors,
                params=params,
                timeout=args.timeout,
                inter_request_delay=args.inter_request_delay,
            )

        print("Starting measurement...")
        print()

        start_time = time.monotonic()
        deadline = start_time + args.seconds
        period = 1.0 / args.target_rate if args.target_rate > 0 else 0.0

        while time.monotonic() < deadline:
            cycle_start = time.monotonic()

            values, statuses, ok_count, fail_count, timeout_count = run_once(
                driver,
                motors=motors,
                params=params,
                timeout=args.timeout,
                inter_request_delay=args.inter_request_delay,
            )

            cycle_elapsed = time.monotonic() - cycle_start

            cycle_times.append(cycle_elapsed)
            ok_counts.append(ok_count)
            fail_counts.append(fail_count)
            timeout_counts.append(timeout_count)

            total_ok += ok_count
            total_fail += fail_count
            total_timeout += timeout_count

            last_values = values
            last_statuses = statuses

            cycle += 1

            if args.print_every > 0 and cycle % args.print_every == 0:
                achieved_so_far = cycle / max(1e-9, (time.monotonic() - start_time))
                print(
                    f"cycle={cycle:05d} "
                    f"dt={cycle_elapsed*1000.0:.2f}ms "
                    f"rate={achieved_so_far:.2f}Hz "
                    f"ok={ok_count}/{request_count_per_cycle} "
                    f"fail={fail_count} timeout={timeout_count}"
                )

            if period > 0:
                sleep_time = period - cycle_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    overrun_count += 1

        elapsed = time.monotonic() - start_time
        achieved_cycle_rate = cycle / elapsed if elapsed > 0 else 0.0
        achieved_request_rate = total_ok / elapsed if elapsed > 0 else 0.0

        stats = driver.get_stats()

        print()
        print("=" * 90)
        print("Read-rate result")
        print("=" * 90)
        print(f"Elapsed:                 {elapsed:.3f} s")
        print(f"Cycles:                  {cycle}")
        print(f"Requests per cycle:       {request_count_per_cycle}")
        print(f"Total OK reads:           {total_ok}")
        print(f"Total failed reads:       {total_fail}")
        print(f"Total TIMEOUT reads:      {total_timeout}")
        print(f"Achieved cycle rate:      {achieved_cycle_rate:.2f} Hz")
        print(f"Achieved OK request rate: {achieved_request_rate:.2f} reads/s")
        print(f"Overrun count:            {overrun_count}")

        if cycle_times:
            avg_dt = statistics.mean(cycle_times)
            med_dt = statistics.median(cycle_times)
            p90_dt = percentile(cycle_times, 90)
            p99_dt = percentile(cycle_times, 99)
            max_dt = max(cycle_times)
            min_dt = min(cycle_times)

            print()
            print("Cycle time:")
            print(f"  min:    {min_dt*1000.0:.2f} ms")
            print(f"  avg:    {avg_dt*1000.0:.2f} ms")
            print(f"  median: {med_dt*1000.0:.2f} ms")
            print(f"  p90:    {p90_dt*1000.0:.2f} ms")
            print(f"  p99:    {p99_dt*1000.0:.2f} ms")
            print(f"  max:    {max_dt*1000.0:.2f} ms")

        print()
        print("Driver stats:")
        print(f"  rx_count:            {stats['rx_count']}")
        print(f"  tx_count:            {stats['tx_count']}")
        print(f"  rx_error_count:      {stats['rx_error_count']}")
        print(f"  tx_error_count:      {stats['tx_error_count']}")
        print(f"  unknown_frame_count: {stats['unknown_frame_count']}")
        print(f"  type2_feedback_count:{stats['type2_feedback_count']}")
        print(f"  param_reply_count:   {stats['param_reply_count']}")

        if args.show_values and last_values is not None and last_statuses is not None:
            print()
            print_last_values(last_values, last_statuses, motors, params)

        print()
        if total_fail == 0:
            print("RESULT: PASS")
        else:
            fail_rate = total_fail / max(1, (total_ok + total_fail))
            if fail_rate < 0.01:
                print("RESULT: PASS_WITH_MINOR_READ_FAILURES")
            else:
                print("RESULT: FAIL")

    except KeyboardInterrupt:
        print()
        print("Interrupted by user.")
        print("RESULT: INTERRUPTED")
        return 130

    except Exception as e:
        print()
        print(f"RESULT: FAIL")
        print(f"ERROR: {e}")
        return 1

    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
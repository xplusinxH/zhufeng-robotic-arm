#!/usr/bin/env python3
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_EFF_MASK = 0x1FFFFFFF

CAN_IFACE_DEFAULT = "can0"
HOST_ID = 0xFD

COMM_MOTION_CONTROL = 1
COMM_FEEDBACK = 2
COMM_ENABLE = 3
COMM_DISABLE = 4
COMM_READ_PARAM = 17
COMM_WRITE_PARAM = 18

RUN_MODE_INDEX = 0x7005
RUN_MODE_MOTION_CONTROL = 0

RS00_LIMITS = {
    "p_min": -12.57,
    "p_max": 12.57,
    "v_min": -33.0,
    "v_max": 33.0,
    "t_min": -14.0,
    "t_max": 14.0,
    "kp_min": 0.0,
    "kp_max": 500.0,
    "kd_min": 0.0,
    "kd_max": 5.0,
}

RS05_LIMITS = {
    "p_min": -12.57,
    "p_max": 12.57,
    "v_min": -50.0,
    "v_max": 50.0,
    "t_min": -5.5,
    "t_max": 5.5,
    "kp_min": 0.0,
    "kp_max": 500.0,
    "kd_min": 0.0,
    "kd_max": 5.0,
}

TYPE1_LIMITS = {
    "RS00": RS00_LIMITS,
    "RS05": RS05_LIMITS,
}

MOTOR_TYPE_BY_ID = {
    1: "RS00",
    2: "RS00",
    3: "RS00",
    4: "RS05",
    5: "RS05",
    6: "RS05",
    7: "RS05",
}


@dataclass
class ParamReply:
    motor_id: int
    index: int
    status: str
    value: Optional[float]
    timestamp: float


@dataclass
class Type2Feedback:
    motor_id: int
    position: float
    velocity: float
    torque: float
    temperature: float
    raw_can_id: int
    raw_data: bytes
    timestamp: float


def build_extended_can_id(comm_type: int, data_area2: int, target_id: int) -> int:
    return ((comm_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (target_id & 0xFF)


def split_extended_can_id(can_id: int) -> Tuple[int, int, int]:
    can_id &= CAN_EFF_MASK
    comm_type = (can_id >> 24) & 0x1F
    data_area2 = (can_id >> 8) & 0xFFFF
    target_id = can_id & 0xFF
    return comm_type, data_area2, target_id


def float_to_uint16(x: float, x_min: float, x_max: float) -> int:
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * 65535.0 / (x_max - x_min))


def uint16_to_float(x_raw: int, x_min: float, x_max: float) -> float:
    return float(x_raw) * (x_max - x_min) / 65535.0 + x_min


def pack_can_frame(can_id_29bit: int, data: bytes) -> bytes:
    if len(data) > 8:
        raise ValueError("CAN data length must be <= 8 bytes")

    can_id = (can_id_29bit & CAN_EFF_MASK) | CAN_EFF_FLAG
    dlc = len(data)
    data_padded = data + bytes(8 - dlc)

    return struct.pack("=IB3x8s", can_id, dlc, data_padded)


def unpack_can_frame(frame: bytes) -> Tuple[int, bytes]:
    can_id_raw, dlc, data = struct.unpack("=IB3x8s", frame)
    is_extended = bool(can_id_raw & CAN_EFF_FLAG)
    if not is_extended:
        raise ValueError("Only extended CAN frames are expected")

    can_id = can_id_raw & CAN_EFF_MASK
    return can_id, data[:dlc]


def make_read_param_data(index: int) -> bytes:
    index_lo = index & 0xFF
    index_hi = (index >> 8) & 0xFF
    return bytes([index_lo, index_hi, 0, 0, 0, 0, 0, 0])


def make_write_param_int_data(index: int, value: int) -> bytes:
    index_lo = index & 0xFF
    index_hi = (index >> 8) & 0xFF
    raw_value = struct.pack("<I", value & 0xFFFFFFFF)
    return bytes([index_lo, index_hi, 0, 0]) + raw_value


def parse_float_from_param_reply(data: bytes) -> float:
    if len(data) != 8:
        raise ValueError(f"Expected 8 bytes, got {len(data)}")
    return struct.unpack("<f", data[4:8])[0]


class SukineeSocketCANDriver:
    """
    Native SocketCAN driver for Sukinee / RobStride-style private protocol.

    This driver does not call cansend or candump.
    It opens one raw CAN socket, starts one background RX thread,
    and caches parameter replies / optional Type2 feedback.
    """

    def __init__(
        self,
        can_iface: str = CAN_IFACE_DEFAULT,
        host_id: int = HOST_ID,
        motor_type_by_id: Optional[Dict[int, str]] = None,
        enable_own_messages: bool = False,
    ):
        self.can_iface = can_iface
        self.host_id = host_id
        self.motor_type_by_id = dict(motor_type_by_id or MOTOR_TYPE_BY_ID)

        self.sock: Optional[socket.socket] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.lock = threading.RLock()
        self.param_condition = threading.Condition(self.lock)

        self.param_replies: Dict[Tuple[int, int], ParamReply] = {}
        self.type2_feedback: Dict[int, Type2Feedback] = {}

        self.rx_count = 0
        self.tx_count = 0
        self.rx_error_count = 0
        self.tx_error_count = 0
        self.unknown_frame_count = 0
        self.start_time = None

        self.enable_own_messages = enable_own_messages

    def open(self):
        if self.sock is not None:
            return

        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)

        # Disable receiving our own transmitted frames by default.
        # This keeps the RX cache from being polluted by outgoing Type1/3/4/17/18 frames.
        if not self.enable_own_messages:
            try:
                self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
            except OSError:
                pass

        self.sock.bind((self.can_iface,))
        self.sock.settimeout(0.05)

        self.stop_event.clear()
        self.start_time = time.monotonic()

        self.rx_thread = threading.Thread(target=self._rx_loop, name="sukinee_socketcan_rx", daemon=True)
        self.rx_thread.start()

    def close(self):
        self.stop_event.set()

        if self.rx_thread is not None:
            self.rx_thread.join(timeout=1.0)
            self.rx_thread = None

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _send_raw(self, can_id_29bit: int, data: bytes):
        if self.sock is None:
            raise RuntimeError("CAN socket is not open")

        frame = pack_can_frame(can_id_29bit, data)

        try:
            self.sock.send(frame)
            with self.lock:
                self.tx_count += 1
        except Exception:
            with self.lock:
                self.tx_error_count += 1
            raise

    def _rx_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.sock is None:
                    return

                frame = self.sock.recv(16)
                can_id, data = unpack_can_frame(frame)
                self._handle_rx_frame(can_id, data)

                with self.lock:
                    self.rx_count += 1

            except socket.timeout:
                continue
            except OSError:
                if not self.stop_event.is_set():
                    with self.lock:
                        self.rx_error_count += 1
                continue
            except Exception:
                with self.lock:
                    self.rx_error_count += 1
                continue

    def _handle_rx_frame(self, can_id: int, data: bytes):
        comm_type, data_area2, target_id = split_extended_can_id(can_id)

        # Type17 parameter response, observed format:
        # success: 0x1100{motor_id}FD
        # fail:    0x1101{motor_id}FD
        # data: index little-endian + ... + float32 little-endian in byte4-7
        if comm_type == COMM_READ_PARAM and target_id == self.host_id:
            self._handle_type17_reply(can_id, data, data_area2)
            return

        # Type2 feedback. Actual availability depends on motor mode / protocol behavior.
        # This parser is included for EDULITE_A3-style cached feedback architecture,
        # but Type17 read_param_float remains the verified path in the current Sukinee stage.
        if comm_type == COMM_FEEDBACK:
            self._handle_type2_feedback(can_id, data, data_area2, target_id)
            return

        with self.lock:
            self.unknown_frame_count += 1

    def _handle_type17_reply(self, can_id: int, data: bytes, data_area2: int):
        if len(data) != 8:
            return

        # data_area2 layout observed:
        #   success data_area2 = 0x00{motor_id}
        #   fail    data_area2 = 0x01{motor_id}
        status_hi = (data_area2 >> 8) & 0xFF
        motor_id = data_area2 & 0xFF

        index = int(data[0]) | (int(data[1]) << 8)

        if status_hi == 0x00:
            status = "OK"
            try:
                value = parse_float_from_param_reply(data)
            except Exception:
                status = "PARSE_ERROR"
                value = None
        elif status_hi == 0x01:
            status = "READ_FAIL"
            value = None
        else:
            status = f"UNKNOWN_STATUS_{status_hi:02X}"
            value = None

        reply = ParamReply(
            motor_id=motor_id,
            index=index,
            status=status,
            value=value,
            timestamp=time.monotonic(),
        )

        with self.param_condition:
            self.param_replies[(motor_id, index)] = reply
            self.param_condition.notify_all()

    def _handle_type2_feedback(self, can_id: int, data: bytes, data_area2: int, target_id: int):
        if len(data) != 8:
            return

        # The exact Type2 feedback format may vary by RobStride firmware / motor family.
        # This parser follows the common MIT-style 16-bit packed layout:
        #   pos_raw, vel_raw, torque_raw, temp_raw = >HHHH
        # motor_id inference is intentionally conservative.
        #
        # In the observed private protocol, Type17 replies put motor_id in data_area2 low byte.
        # Some Type2 frames may put motor_id in target_id or data_area2 low byte.
        # We choose a robust heuristic and keep raw data for later verification.
        possible_motor_ids = []
        if target_id in self.motor_type_by_id:
            possible_motor_ids.append(target_id)
        low = data_area2 & 0xFF
        if low in self.motor_type_by_id:
            possible_motor_ids.append(low)

        if not possible_motor_ids:
            with self.lock:
                self.unknown_frame_count += 1
            return

        motor_id = possible_motor_ids[0]
        motor_type = self.motor_type_by_id[motor_id]
        limits = TYPE1_LIMITS[motor_type]

        try:
            pos_raw, vel_raw, torque_raw, temp_raw = struct.unpack(">HHHH", data)
            position = uint16_to_float(pos_raw, limits["p_min"], limits["p_max"])
            velocity = uint16_to_float(vel_raw, limits["v_min"], limits["v_max"])
            torque = uint16_to_float(torque_raw, limits["t_min"], limits["t_max"])
            temperature = float(temp_raw) / 10.0
        except Exception:
            with self.lock:
                self.unknown_frame_count += 1
            return

        fb = Type2Feedback(
            motor_id=motor_id,
            position=position,
            velocity=velocity,
            torque=torque,
            temperature=temperature,
            raw_can_id=can_id,
            raw_data=data,
            timestamp=time.monotonic(),
        )

        with self.lock:
            self.type2_feedback[motor_id] = fb

    def read_param_float(self, motor_id: int, index: int, timeout: float = 0.5) -> Tuple[str, Optional[float]]:
        if self.sock is None:
            raise RuntimeError("CAN socket is not open")

        key = (motor_id, index)

        with self.param_condition:
            old_timestamp = self.param_replies[key].timestamp if key in self.param_replies else None

        can_id = build_extended_can_id(COMM_READ_PARAM, self.host_id, motor_id)
        data = make_read_param_data(index)

        self._send_raw(can_id, data)

        deadline = time.monotonic() + timeout

        with self.param_condition:
            while time.monotonic() < deadline:
                reply = self.param_replies.get(key)

                if reply is not None:
                    if old_timestamp is None or reply.timestamp > old_timestamp:
                        return reply.status, reply.value

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                self.param_condition.wait(timeout=remaining)

        return "TIMEOUT", None

    def read_many_params_float(
        self,
        motor_ids,
        params,
        timeout: float = 0.5,
        inter_request_delay: float = 0.002,
    ):
        values = {}
        statuses = {}

        for motor_id in motor_ids:
            values[motor_id] = {}
            statuses[motor_id] = {}

            for index, name in params:
                status, value = self.read_param_float(motor_id, index, timeout=timeout)
                statuses[motor_id][name] = status
                values[motor_id][name] = value
                time.sleep(inter_request_delay)

        return values, statuses

    def get_latest_type2_feedback(self, motor_id: int, max_age: Optional[float] = None) -> Optional[Type2Feedback]:
        with self.lock:
            fb = self.type2_feedback.get(motor_id)

        if fb is None:
            return None

        if max_age is not None and time.monotonic() - fb.timestamp > max_age:
            return None

        return fb

    def get_stats(self):
        with self.lock:
            elapsed = None if self.start_time is None else time.monotonic() - self.start_time
            return {
                "elapsed": elapsed,
                "rx_count": self.rx_count,
                "tx_count": self.tx_count,
                "rx_error_count": self.rx_error_count,
                "tx_error_count": self.tx_error_count,
                "unknown_frame_count": self.unknown_frame_count,
                "type2_feedback_count": len(self.type2_feedback),
                "param_reply_count": len(self.param_replies),
            }

    def send_disable(self, motor_id: int, clear_fault: bool = False):
        can_id = build_extended_can_id(COMM_DISABLE, self.host_id, motor_id)
        first_byte = 1 if clear_fault else 0
        data = bytes([first_byte]) + bytes(7)
        self._send_raw(can_id, data)

    def send_enable(self, motor_id: int):
        can_id = build_extended_can_id(COMM_ENABLE, self.host_id, motor_id)
        data = bytes(8)
        self._send_raw(can_id, data)

    def send_write_param_int(self, motor_id: int, index: int, value: int):
        can_id = build_extended_can_id(COMM_WRITE_PARAM, self.host_id, motor_id)
        data = make_write_param_int_data(index, value)
        self._send_raw(can_id, data)

    def send_set_motion_mode(self, motor_id: int):
        self.send_write_param_int(motor_id, RUN_MODE_INDEX, RUN_MODE_MOTION_CONTROL)

    def send_motion_control(
        self,
        motor_id: int,
        position: float,
        velocity: float,
        kp: float,
        kd: float,
        torque: float,
    ):
        motor_type = self.motor_type_by_id[motor_id]
        limits = TYPE1_LIMITS[motor_type]

        pos_raw = float_to_uint16(position, limits["p_min"], limits["p_max"])
        vel_raw = float_to_uint16(velocity, limits["v_min"], limits["v_max"])
        kp_raw = float_to_uint16(kp, limits["kp_min"], limits["kp_max"])
        kd_raw = float_to_uint16(kd, limits["kd_min"], limits["kd_max"])
        torque_raw = float_to_uint16(torque, limits["t_min"], limits["t_max"])

        can_id = build_extended_can_id(COMM_MOTION_CONTROL, torque_raw, motor_id)
        data = struct.pack(">HHHH", pos_raw, vel_raw, kp_raw, kd_raw)

        self._send_raw(can_id, data)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Native SocketCAN driver smoke test.")
    parser.add_argument("--can", default=CAN_IFACE_DEFAULT)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--read-all", action="store_true")
    args = parser.parse_args()

    driver = SukineeSocketCANDriver(args.can)

    try:
        print("Opening native SocketCAN driver...")
        driver.open()

        print("Driver opened.")
        print("No Type1 / Type3 / Type4 / Type18 command will be sent by this smoke test.")
        print("Only Type17 read_param is used if --read-all is provided.")
        print()

        if args.read_all:
            params = [
                (0x7019, "pos"),
                (0x701A, "iqf"),
                (0x701B, "vel"),
                (0x701C, "vbus"),
            ]

            values, statuses = driver.read_many_params_float(
                motor_ids=[1, 2, 3, 4, 5, 6, 7],
                params=params,
                timeout=0.5,
                inter_request_delay=0.003,
            )

            for mid in [1, 2, 3, 4, 5, 6, 7]:
                print(f"Joint{mid}:")
                for _idx, name in params:
                    st = statuses[mid][name]
                    val = values[mid][name]
                    if st == "OK":
                        print(f"  {name}: {val:+.6f}")
                    else:
                        print(f"  {name}: {st}")
                print()

        print(f"Listening for {args.seconds:.1f} seconds...")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(0.5)
            stats = driver.get_stats()
            print(
                f"stats: rx={stats['rx_count']} tx={stats['tx_count']} "
                f"type2={stats['type2_feedback_count']} "
                f"param={stats['param_reply_count']} "
                f"unknown={stats['unknown_frame_count']} "
                f"rx_err={stats['rx_error_count']} tx_err={stats['tx_error_count']}"
            )

        print()
        print("RESULT: PASS")

    except Exception as e:
        print()
        print(f"RESULT: FAIL: {e}")
        return 1

    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
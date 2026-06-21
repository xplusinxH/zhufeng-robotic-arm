#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rotation


JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
ALLOWED_RELATIVE_POSE_SOURCES = {"same_frame", "cached_base_ref", "mixed"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_no}")
            records.append(obj)
    return records


def load_offsets_current_format(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    sign_map = obj.get("motor_to_urdf_sign")
    zero_map = obj.get("motor_pos_at_urdf_zero")

    if not isinstance(sign_map, dict):
        raise ValueError(f"offset file missing required dict field motor_to_urdf_sign: {path}")
    if not isinstance(zero_map, dict):
        raise ValueError(f"offset file missing required dict field motor_pos_at_urdf_zero: {path}")

    for name in JOINT_NAMES:
        if name not in sign_map:
            raise ValueError(f"offset file missing motor_to_urdf_sign[{name!r}]: {path}")
        if name not in zero_map:
            raise ValueError(f"offset file missing motor_pos_at_urdf_zero[{name!r}]: {path}")
        float(sign_map[name])
        float(zero_map[name])

    return obj


def index_by_sample_id(records: Iterable[Dict[str, Any]], source_name: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{source_name} record missing non-empty sample_id")
        if sid in out:
            raise ValueError(f"duplicate sample_id in {source_name}: {sid}")
        out[sid] = rec
    return out


def get_position_xyz(pos: Any, field_name: str) -> np.ndarray:
    if isinstance(pos, dict):
        vals = [pos.get("x"), pos.get("y"), pos.get("z")]
    elif isinstance(pos, list) and len(pos) == 3:
        vals = pos
    else:
        raise ValueError(f"{field_name} must be xyz dict or length-3 list")

    arr = np.array([float(v) for v in vals], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains non-finite values: {arr}")
    return arr


def get_quat_xyzw(quat: Any, field_name: str) -> np.ndarray:
    if isinstance(quat, dict):
        vals = [quat.get("x"), quat.get("y"), quat.get("z"), quat.get("w")]
    elif isinstance(quat, list) and len(quat) == 4:
        vals = quat
    else:
        raise ValueError(f"{field_name} must be xyzw dict or length-4 list")

    arr = np.array([float(v) for v in vals], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains non-finite values: {arr}")
    n = float(np.linalg.norm(arr))
    if n <= 0:
        raise ValueError(f"{field_name} quaternion norm is zero")
    return arr / n


def make_T(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def vision_T_R_T(record: Dict[str, Any], translation_scale: float) -> np.ndarray:
    if record.get("valid") is False:
        raise ValueError(f"vision sample {record.get('sample_id')} is invalid")

    pose = record.get("T_R_T")
    if not isinstance(pose, dict):
        raise ValueError(f"vision sample {record.get('sample_id')} missing T_R_T")

    pos_raw = get_position_xyz(pose.get("position"), "T_R_T.position")
    quat_xyzw = get_quat_xyzw(pose.get("orientation_xyzw"), "T_R_T.orientation_xyzw")

    pos_scaled = pos_raw * float(translation_scale)
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    return make_T(rot, pos_scaled)


def params6_to_T(params6: np.ndarray) -> np.ndarray:
    params6 = np.asarray(params6, dtype=float)
    if params6.shape != (6,):
        raise ValueError(f"params6 must have shape (6,), got {params6.shape}")
    t = params6[:3]
    rotvec = params6[3:]
    Rm = Rotation.from_rotvec(rotvec).as_matrix()
    return make_T(Rm, t)


def T_to_params6(T: np.ndarray) -> np.ndarray:
    t = np.asarray(T[:3, 3], dtype=float)
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([t, rotvec])


def T_to_xyz_quat_xyzw(T: np.ndarray) -> Tuple[List[float], List[float]]:
    xyz = [float(v) for v in T[:3, 3]]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    quat = quat / np.linalg.norm(quat)
    return xyz, [float(v) for v in quat]


def transform_error(T_left: np.ndarray, T_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    T_err = inv_T(T_left) @ T_right
    trans = T_err[:3, 3]
    rotvec = Rotation.from_matrix(T_err[:3, :3]).as_rotvec()
    return trans, rotvec, float(np.linalg.norm(trans)), float(np.linalg.norm(rotvec))


def source_weight(source: str, cached_weight: float, mixed_weight: float) -> float:
    if source == "same_frame":
        return 1.0
    if source == "cached_base_ref":
        return float(cached_weight)
    if source == "mixed":
        return float(mixed_weight)
    return 0.0


def parse_joint_delta_sigma(items: Iterable[str]) -> Dict[str, float]:
    """Parse items like Joint2:0.01 into a per-joint delta-q prior sigma map.

    A smaller sigma means a stronger prior keeping that joint's delta_q near zero.
    Joints not listed here have no delta prior and are only limited by delta_bound_rad.
    """
    out: Dict[str, float] = {}
    valid = set(JOINT_NAMES)
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"bad --joint-delta-sigma item {item!r}; expected format like Joint2:0.01"
            )
        name, value = item.split(":", 1)
        name = name.strip()
        if name not in valid:
            raise ValueError(f"unknown joint in --joint-delta-sigma: {name!r}; valid joints: {JOINT_NAMES}")
        sigma = float(value)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma for {name} must be positive finite, got {value!r}")
        out[name] = sigma
    return out


class PinocchioFk:
    def __init__(self, urdf_path: Path, base_frame: str, tool_frame: str):
        try:
            import pinocchio as pin  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Python module pinocchio is not available. In your ROS/Pinocchio environment, "
                "source the workspace or install the Pinocchio Python binding first."
            ) from exc

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        self.base_frame = base_frame
        self.tool_frame = tool_frame
        self.base_frame_id = self.model.getFrameId(base_frame)
        self.tool_frame_id = self.model.getFrameId(tool_frame)

        if self.base_frame_id >= len(self.model.frames):
            raise RuntimeError(f"Pinocchio frame not found: {base_frame}")
        if self.tool_frame_id >= len(self.model.frames):
            raise RuntimeError(f"Pinocchio frame not found: {tool_frame}")

        self.joint_index: Dict[str, Tuple[int, int, int]] = {}
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise RuntimeError(f"Pinocchio joint not found in URDF: {name}")
            self.joint_index[name] = (jid, self.model.idx_qs[jid], self.model.idx_vs[jid])

    def build_q(self, q_joint: Dict[str, float]) -> np.ndarray:
        q = self.pin.neutral(self.model)
        for name in JOINT_NAMES:
            jid, idx_q, _idx_v = self.joint_index[name]
            nq = self.model.nqs[jid]
            nv = self.model.nvs[jid]
            theta = float(q_joint[name])

            if nq == 1 and nv == 1:
                q[idx_q] = theta
            elif nq == 2 and nv == 1:
                q[idx_q] = math.cos(theta)
                q[idx_q + 1] = math.sin(theta)
            else:
                raise RuntimeError(
                    f"{name} has nq={nq}, nv={nv}; expected revolute nq=1,nv=1 or continuous nq=2,nv=1"
                )
        return q

    def T_B_E(self, q_joint: Dict[str, float]) -> np.ndarray:
        q = self.build_q(q_joint)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

        T_o_b = self.data.oMf[self.base_frame_id]
        T_o_e = self.data.oMf[self.tool_frame_id]
        T_b_e = T_o_b.inverse() * T_o_e

        return make_T(np.array(T_b_e.rotation), np.array(T_b_e.translation).reshape(3))


def build_pairs(
    vision_path: Path,
    robot_path: Path,
    *,
    vision_translation_scale: float,
    cached_weight: float,
    mixed_weight: float,
    exclude_sample_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    vision_records = index_by_sample_id(load_jsonl(vision_path), "vision")
    robot_records = index_by_sample_id(load_jsonl(robot_path), "robot")

    common_ids = sorted(set(vision_records.keys()) & set(robot_records.keys()))
    if not common_ids:
        raise RuntimeError("no common sample_id between vision and robot JSONL")

    exclude_ids = {str(x).strip() for x in exclude_sample_ids if str(x).strip()}

    missing_v = sorted(set(robot_records.keys()) - set(vision_records.keys()))
    missing_r = sorted(set(vision_records.keys()) - set(robot_records.keys()))
    if missing_v or missing_r:
        raise RuntimeError(
            f"sample_id mismatch: missing vision for {missing_v[:5]}, missing robot for {missing_r[:5]}"
        )

    pairs: List[Dict[str, Any]] = []
    for sid in common_ids:
        if sid in exclude_ids:
            continue

        vrec = vision_records[sid]
        rrec = robot_records[sid]

        if vrec.get("valid") is not True:
            continue
        if rrec.get("valid") is not True:
            continue

        source = vrec.get("relative_pose_source", "unknown")
        if source not in ALLOWED_RELATIVE_POSE_SOURCES:
            continue

        q_obj = rrec.get("q_urdf_current_rad")
        if not isinstance(q_obj, dict):
            raise ValueError(f"robot sample {sid} missing q_urdf_current_rad")

        q_joint: Dict[str, float] = {}
        for name in JOINT_NAMES:
            if name not in q_obj:
                raise ValueError(f"robot sample {sid} missing q_urdf_current_rad[{name}]")
            q_joint[name] = float(q_obj[name])

        T_rt = vision_T_R_T(vrec, vision_translation_scale)
        w = source_weight(source, cached_weight, mixed_weight)
        if w <= 0.0:
            continue

        pairs.append(
            {
                "sample_id": sid,
                "relative_pose_source": source,
                "base_ref_source": vrec.get("base_ref_source"),
                "weight": w,
                "q_joint": q_joint,
                "T_R_T": T_rt,
                "vision_raw": vrec,
                "robot_raw": rrec,
            }
        )

    if len(pairs) < 12:
        raise RuntimeError(f"too few usable pairs after filtering: {len(pairs)}")

    return pairs


def q_with_delta(q_joint: Dict[str, float], delta_q: np.ndarray) -> Dict[str, float]:
    return {name: float(q_joint[name] + delta_q[i]) for i, name in enumerate(JOINT_NAMES)}


def residual_vector(
    variables: np.ndarray,
    *,
    pairs: List[Dict[str, Any]],
    fk: PinocchioFk,
    sigma_t: float,
    sigma_r: float,
    fit_delta_q: bool,
    delta_prior_sigmas: Dict[str, float] | None = None,
) -> np.ndarray:
    variables = np.asarray(variables, dtype=float)

    if fit_delta_q:
        delta_q = variables[:6]
        ext = variables[6:]
    else:
        delta_q = np.zeros(6, dtype=float)
        ext = variables

    T_B_R = params6_to_T(ext[:6])
    T_E_T = params6_to_T(ext[6:12])

    residuals: List[float] = []
    for pair in pairs:
        T_R_T = pair["T_R_T"]
        qj = q_with_delta(pair["q_joint"], delta_q)
        T_B_E = fk.T_B_E(qj)

        T_left = T_B_R @ T_R_T
        T_right = T_B_E @ T_E_T
        trans, rotvec, _tn, _rn = transform_error(T_left, T_right)

        w = math.sqrt(float(pair["weight"]))
        residuals.extend((w * trans / float(sigma_t)).tolist())
        residuals.extend((w * rotvec / float(sigma_r)).tolist())

    if fit_delta_q and delta_prior_sigmas:
        # Delta-q prior residuals: keep selected software joint-offset corrections near zero.
        # Example: Joint2:0.01 adds residual delta_q[Joint2] / 0.01.
        for i, name in enumerate(JOINT_NAMES):
            sigma = delta_prior_sigmas.get(name)
            if sigma is not None:
                residuals.append(float(delta_q[i]) / float(sigma))

    return np.array(residuals, dtype=float)


def mean_transform(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4, dtype=float)
    translations = np.array([T[:3, 3] for T in transforms], dtype=float)
    rotations = Rotation.from_matrix([T[:3, :3] for T in transforms])
    try:
        r_mean = rotations.mean()
    except Exception:
        r_mean = rotations[0]
    return make_T(r_mean.as_matrix(), np.mean(translations, axis=0))


def make_initial_ext_seeds(pairs: List[Dict[str, Any]], fk: PinocchioFk) -> List[Tuple[str, np.ndarray]]:
    seeds: List[Tuple[str, np.ndarray]] = []

    seeds.append(("identity", np.zeros(12, dtype=float)))

    base_candidates: List[np.ndarray] = []
    tool_candidates: List[np.ndarray] = []
    for pair in pairs:
        T_B_E = fk.T_B_E(pair["q_joint"])
        T_R_T = pair["T_R_T"]

        # 假设 T_E_T = I，则 T_B_R ≈ T_B_E * inv(T_R_T)
        base_candidates.append(T_B_E @ inv_T(T_R_T))

        # 假设 T_B_R = I，则 T_E_T ≈ inv(T_B_E) * T_R_T
        tool_candidates.append(inv_T(T_B_E) @ T_R_T)

    T_B_R0 = mean_transform(base_candidates)
    seeds.append(("base_from_tool_identity", np.concatenate([T_to_params6(T_B_R0), np.zeros(6)])))

    T_E_T0 = mean_transform(tool_candidates)
    seeds.append(("tool_from_base_identity", np.concatenate([np.zeros(6), T_to_params6(T_E_T0)])))

    return seeds


def optimize_extrinsics_only(
    pairs: List[Dict[str, Any]],
    fk: PinocchioFk,
    *,
    sigma_t: float,
    sigma_r: float,
    max_nfev: int,
    loss: str,
) -> Tuple[str, Any]:
    seeds = make_initial_ext_seeds(pairs, fk)
    lower = np.array([-2.0, -2.0, -2.0, -math.pi, -math.pi, -math.pi] * 2, dtype=float)
    upper = np.array([2.0, 2.0, 2.0, math.pi, math.pi, math.pi] * 2, dtype=float)

    best_name = ""
    best_result = None
    for name, x0 in seeds:
        result = least_squares(
            residual_vector,
            x0=x0,
            bounds=(lower, upper),
            args=(),
            kwargs={
                "pairs": pairs,
                "fk": fk,
                "sigma_t": sigma_t,
                "sigma_r": sigma_r,
                "fit_delta_q": False,
            },
            loss=loss,
            max_nfev=max_nfev,
            verbose=0,
        )
        if best_result is None or result.cost < best_result.cost:
            best_name = name
            best_result = result

    if best_result is None:
        raise RuntimeError("extrinsics-only optimization did not run")
    return best_name, best_result


def optimize_full(
    pairs: List[Dict[str, Any]],
    fk: PinocchioFk,
    ext0: np.ndarray,
    *,
    sigma_t: float,
    sigma_r: float,
    delta_bound_rad: float,
    delta_prior_sigmas: Dict[str, float],
    max_nfev: int,
    loss: str,
) -> Any:
    x0 = np.concatenate([np.zeros(6, dtype=float), ext0])

    lower_delta = np.full(6, -float(delta_bound_rad), dtype=float)
    upper_delta = np.full(6, float(delta_bound_rad), dtype=float)
    lower_ext = np.array([-2.0, -2.0, -2.0, -math.pi, -math.pi, -math.pi] * 2, dtype=float)
    upper_ext = np.array([2.0, 2.0, 2.0, math.pi, math.pi, math.pi] * 2, dtype=float)

    return least_squares(
        residual_vector,
        x0=x0,
        bounds=(np.concatenate([lower_delta, lower_ext]), np.concatenate([upper_delta, upper_ext])),
        args=(),
        kwargs={
            "pairs": pairs,
            "fk": fk,
            "sigma_t": sigma_t,
            "sigma_r": sigma_r,
            "fit_delta_q": True,
            "delta_prior_sigmas": delta_prior_sigmas,
        },
        loss=loss,
        max_nfev=max_nfev,
        verbose=1,
    )


def per_sample_errors(
    variables: np.ndarray,
    *,
    pairs: List[Dict[str, Any]],
    fk: PinocchioFk,
    fit_delta_q: bool,
) -> Dict[str, Dict[str, float]]:
    if fit_delta_q:
        delta_q = variables[:6]
        ext = variables[6:]
    else:
        delta_q = np.zeros(6, dtype=float)
        ext = variables

    T_B_R = params6_to_T(ext[:6])
    T_E_T = params6_to_T(ext[6:12])

    out: Dict[str, Dict[str, float]] = {}
    for pair in pairs:
        qj = q_with_delta(pair["q_joint"], delta_q)
        T_B_E = fk.T_B_E(qj)
        T_left = T_B_R @ pair["T_R_T"]
        T_right = T_B_E @ T_E_T
        _trans, _rotvec, tn, rn = transform_error(T_left, T_right)
        out[pair["sample_id"]] = {
            "translation_error_m": tn,
            "rotation_error_rad": rn,
        }
    return out


def stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "rmse": math.nan, "median": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "count": int(arr.size),
        "rmse": float(math.sqrt(np.mean(arr * arr))),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def grouped_stats(pairs: List[Dict[str, Any]], err: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    groups: Dict[str, List[str]] = {"overall": [p["sample_id"] for p in pairs]}
    for p in pairs:
        groups.setdefault(p["relative_pose_source"], []).append(p["sample_id"])

    out: Dict[str, Any] = {}
    for group_name, ids in groups.items():
        trans = [err[sid]["translation_error_m"] for sid in ids]
        rot = [err[sid]["rotation_error_rad"] for sid in ids]
        out[group_name] = {
            "translation_m": stats(trans),
            "rotation_rad": stats(rot),
        }
    return out


def write_yaml_transform(path: Path, *, parent: str, child: str, T: np.ndarray) -> None:
    xyz, quat = T_to_xyz_quat_xyzw(T)
    lines = [
        f"parent_frame: {parent}",
        f"child_frame: {child}",
        "translation:",
        f"  x: {xyz[0]:.12g}",
        f"  y: {xyz[1]:.12g}",
        f"  z: {xyz[2]:.12g}",
        "orientation_xyzw:",
        f"  x: {quat[0]:.12g}",
        f"  y: {quat[1]:.12g}",
        f"  z: {quat[2]:.12g}",
        f"  w: {quat[3]:.12g}",
        "meaning: generated by offline Gate5 calibration; review before use",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_dir: Path,
    overwrite: bool,
    args: argparse.Namespace,
    pairs: List[Dict[str, Any]],
    offsets_raw: Dict[str, Any],
    ext_only_seed_name: str,
    ext_result: Any,
    full_result: Any,
    before_errors: Dict[str, Dict[str, float]],
    after_errors: Dict[str, Dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "offset": output_dir / "sukinee_motor_to_urdf_offsets_calibrated.json",
        "base_yaml": output_dir / "base_link_to_tag_base_ref.yaml",
        "tool_yaml": output_dir / "tool0_to_tag_tool0.yaml",
        "csv": output_dir / "residuals_train.csv",
        "report": output_dir / "calibration_report_before_after.txt",
    }
    if not overwrite:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(
                "output files already exist; remove them manually or pass --overwrite:\n" + "\n".join(existing)
            )

    delta_q = full_result.x[:6]
    ext_after = full_result.x[6:]
    T_B_R = params6_to_T(ext_after[:6])
    T_E_T = params6_to_T(ext_after[6:12])

    sign_map = offsets_raw["motor_to_urdf_sign"]
    zero_old = offsets_raw["motor_pos_at_urdf_zero"]

    zero_new: Dict[str, float] = {}
    delta_q_map: Dict[str, float] = {}
    for i, name in enumerate(JOINT_NAMES):
        sign = float(sign_map[name])
        dq = float(delta_q[i])
        delta_q_map[name] = dq
        zero_new[name] = float(zero_old[name]) - sign * dq

    calibrated = dict(offsets_raw)
    calibrated["motor_pos_at_urdf_zero_before_gate5"] = {name: float(zero_old[name]) for name in JOINT_NAMES}
    calibrated["motor_pos_at_urdf_zero"] = zero_new
    calibrated["gate5_calibration_metadata"] = {
        "created_at_utc": now_iso(),
        "script": "sukinee_optimize_zero_offsets_tooltag_base_ref.py",
        "source_offset_file": str(args.offset_file),
        "vision_samples": str(args.vision),
        "robot_samples": str(args.robot),
        "urdf": str(args.urdf),
        "sample_count": len(pairs),
        "excluded_sample_ids": list(args.exclude_sample_ids),
        "joint_delta_sigma": dict(args.joint_delta_sigma_map),
        "delta_prior_enabled": bool(args.joint_delta_sigma_map),
        "configured_tag_size_m": args.configured_tag_size_m,
        "measured_tag_size_m": args.measured_tag_size_m,
        "vision_translation_scale": args.vision_translation_scale,
        "delta_q_rad": delta_q_map,
        "formula": "zero_new[j] = zero_old[j] - sign[j] * delta_q[j]",
        "safety": {
            "software_only": True,
            "type6_set_zero": False,
            "save_motor_parameters": False,
            "moveit_real_execution": False,
        },
    }
    paths["offset"].write_text(json.dumps(calibrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_yaml_transform(paths["base_yaml"], parent=args.base_frame, child="tag_base_ref", T=T_B_R)
    write_yaml_transform(paths["tool_yaml"], parent=args.tool_frame, child="tag_tool0", T=T_E_T)

    with paths["csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "relative_pose_source",
                "base_ref_source",
                "weight",
                "before_translation_error_m",
                "before_rotation_error_rad",
                "after_translation_error_m",
                "after_rotation_error_rad",
                "improved_translation_m",
                "improved_rotation_rad",
            ],
        )
        writer.writeheader()
        for p in pairs:
            sid = p["sample_id"]
            b = before_errors[sid]
            a = after_errors[sid]
            writer.writerow(
                {
                    "sample_id": sid,
                    "relative_pose_source": p["relative_pose_source"],
                    "base_ref_source": p.get("base_ref_source"),
                    "weight": p["weight"],
                    "before_translation_error_m": b["translation_error_m"],
                    "before_rotation_error_rad": b["rotation_error_rad"],
                    "after_translation_error_m": a["translation_error_m"],
                    "after_rotation_error_rad": a["rotation_error_rad"],
                    "improved_translation_m": b["translation_error_m"] - a["translation_error_m"],
                    "improved_rotation_rad": b["rotation_error_rad"] - a["rotation_error_rad"],
                }
            )

    before_stats = grouped_stats(pairs, before_errors)
    after_stats = grouped_stats(pairs, after_errors)

    lines: List[str] = []
    lines.append("Sukinee Gate5 calibration report")
    lines.append("================================")
    lines.append(f"created_at_utc: {now_iso()}")
    lines.append(f"sample_count: {len(pairs)}")
    lines.append(f"excluded_sample_ids: {list(args.exclude_sample_ids)}")
    lines.append(f"joint_delta_sigma: {dict(args.joint_delta_sigma_map)}")
    lines.append(f"delta_prior_enabled: {bool(args.joint_delta_sigma_map)}")
    lines.append(f"vision_translation_scale: {args.vision_translation_scale}")
    lines.append(f"configured_tag_size_m: {args.configured_tag_size_m}")
    lines.append(f"measured_tag_size_m: {args.measured_tag_size_m}")
    lines.append(f"cached_weight: {args.cached_weight}")
    lines.append(f"mixed_weight: {args.mixed_weight}")
    lines.append("")
    lines.append("Optimization")
    lines.append("------------")
    lines.append(f"extrinsics_only_initial_seed_selected: {ext_only_seed_name}")
    lines.append(f"extrinsics_only_success: {ext_result.success}")
    lines.append(f"extrinsics_only_cost: {ext_result.cost:.12g}")
    lines.append(f"full_success: {full_result.success}")
    lines.append(f"full_cost: {full_result.cost:.12g}")
    lines.append(f"full_message: {full_result.message}")
    lines.append("")
    lines.append("Delta q result")
    lines.append("--------------")
    for name in JOINT_NAMES:
        sigma = args.joint_delta_sigma_map.get(name)
        prior_residual = None if sigma is None else delta_q_map[name] / float(sigma)
        if sigma is None:
            lines.append(f"{name}: delta_q_rad={delta_q_map[name]: .12g}, zero_old={float(zero_old[name]): .12g}, zero_new={zero_new[name]: .12g}, delta_prior_sigma=None")
        else:
            lines.append(f"{name}: delta_q_rad={delta_q_map[name]: .12g}, zero_old={float(zero_old[name]): .12g}, zero_new={zero_new[name]: .12g}, delta_prior_sigma={sigma:.12g}, delta_prior_residual={prior_residual:.12g}")
    lines.append("")
    lines.append("Before/after residual stats")
    lines.append("---------------------------")
    for group_name in sorted(after_stats.keys()):
        lines.append(f"[{group_name}]")
        lines.append(f"  before.translation_m: {before_stats[group_name]['translation_m']}")
        lines.append(f"  after.translation_m : {after_stats[group_name]['translation_m']}")
        lines.append(f"  before.rotation_rad : {before_stats[group_name]['rotation_rad']}")
        lines.append(f"  after.rotation_rad  : {after_stats[group_name]['rotation_rad']}")
    lines.append("")
    lines.append("Largest after-translation residual samples")
    lines.append("------------------------------------------")
    worst = sorted(pairs, key=lambda p: after_errors[p["sample_id"]]["translation_error_m"], reverse=True)[:10]
    for p in worst:
        sid = p["sample_id"]
        lines.append(
            f"{sid}: source={p['relative_pose_source']}, "
            f"after_translation_m={after_errors[sid]['translation_error_m']:.12g}, "
            f"after_rotation_rad={after_errors[sid]['rotation_error_rad']:.12g}"
        )
    lines.append("")
    lines.append("Safety boundary")
    lines.append("---------------")
    lines.append("software_only: true")
    lines.append("type6_set_zero: false")
    lines.append("save_motor_parameters: false")
    lines.append("moveit_real_execution: false")
    lines.append("can0_required: false")

    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[saved]")
    for key, p in paths.items():
        print(f"  {key}: {p}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Offline Gate5 joint-offset + tag extrinsic calibration for Sukinee. No CAN, no ROS node, no real control."
    )
    default_run = Path("/home/zzj/sukinee_ws/vision_calibration/data/run_100_20260619_012357")
    ap.add_argument("--vision", type=Path, default=default_run / "vision_samples.jsonl")
    ap.add_argument("--robot", type=Path, default=default_run / "robot_samples.jsonl")
    ap.add_argument("--urdf", type=Path, default=Path("/home/zzj/sukinee_ws/src/sukinee_urdf/urdf/sukinee_urdf.urdf"))
    ap.add_argument("--offset-file", type=Path, default=Path("/home/zzj/sukinee_ws/sukinee_motor_to_urdf_offsets.json"))
    ap.add_argument("--output-dir", type=Path, default=default_run / "gate5_result_77mm")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--tool-frame", default="tool0")

    ap.add_argument("--configured-tag-size-m", type=float, default=0.08)
    ap.add_argument("--measured-tag-size-m", type=float, default=0.077)
    ap.add_argument("--vision-translation-scale", type=float, default=0.9625)

    ap.add_argument("--cached-weight", type=float, default=0.5)
    ap.add_argument("--mixed-weight", type=float, default=0.7)
    ap.add_argument("--sigma-t", type=float, default=0.01, help="translation normalization in meters")
    ap.add_argument("--sigma-r", type=float, default=0.05, help="rotation normalization in radians")
    ap.add_argument("--delta-bound-rad", type=float, default=0.35)
    ap.add_argument("--max-nfev-ext", type=int, default=200)
    ap.add_argument("--max-nfev-full", type=int, default=500)
    ap.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    ap.add_argument(
        "--exclude-sample-ids",
        nargs="*",
        default=[],
        help="sample IDs to exclude from optimization, for example: --exclude-sample-ids P0021 P0008",
    )
    ap.add_argument(
        "--joint-delta-sigma",
        nargs="*",
        default=[],
        help=(
            "per-joint delta-q prior sigma in radians, for example: "
            "--joint-delta-sigma Joint2:0.01. "
            "Listed joints get residual delta_q/sigma; unlisted joints have no delta prior."
        ),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.joint_delta_sigma_map = parse_joint_delta_sigma(args.joint_delta_sigma)
    return args


def main() -> None:
    args = parse_args()

    if not args.vision.exists():
        raise FileNotFoundError(args.vision)
    if not args.robot.exists():
        raise FileNotFoundError(args.robot)
    if not args.urdf.exists():
        raise FileNotFoundError(args.urdf)
    if not args.offset_file.exists():
        raise FileNotFoundError(args.offset_file)

    expected_scale = args.measured_tag_size_m / args.configured_tag_size_m
    if abs(expected_scale - args.vision_translation_scale) > 1e-9:
        print(
            "[warning] vision_translation_scale does not equal measured/configured: "
            f"{args.vision_translation_scale} vs {expected_scale}"
        )

    offsets_raw = load_offsets_current_format(args.offset_file)
    pairs = build_pairs(
        args.vision,
        args.robot,
        vision_translation_scale=args.vision_translation_scale,
        cached_weight=args.cached_weight,
        mixed_weight=args.mixed_weight,
        exclude_sample_ids=args.exclude_sample_ids,
    )

    source_counts: Dict[str, int] = {}
    for p in pairs:
        source_counts[p["relative_pose_source"]] = source_counts.get(p["relative_pose_source"], 0) + 1

    print(f"[pairs] usable={len(pairs)} source_counts={source_counts}")
    if args.exclude_sample_ids:
        print(f"[excluded] {args.exclude_sample_ids}")
    if args.joint_delta_sigma_map:
        print(f"[delta prior] {args.joint_delta_sigma_map}")
    print(f"[vision scale] {args.vision_translation_scale} = {args.measured_tag_size_m} / {args.configured_tag_size_m}")

    fk = PinocchioFk(args.urdf, args.base_frame, args.tool_frame)

    seed_name, ext_result = optimize_extrinsics_only(
        pairs,
        fk,
        sigma_t=args.sigma_t,
        sigma_r=args.sigma_r,
        max_nfev=args.max_nfev_ext,
        loss=args.loss,
    )
    print(f"[extrinsics-only] seed={seed_name} success={ext_result.success} cost={ext_result.cost:.12g}")

    full_result = optimize_full(
        pairs,
        fk,
        ext_result.x,
        sigma_t=args.sigma_t,
        sigma_r=args.sigma_r,
        delta_bound_rad=args.delta_bound_rad,
        delta_prior_sigmas=args.joint_delta_sigma_map,
        max_nfev=args.max_nfev_full,
        loss=args.loss,
    )
    print(f"[full] success={full_result.success} cost={full_result.cost:.12g}")

    before_errors = per_sample_errors(ext_result.x, pairs=pairs, fk=fk, fit_delta_q=False)
    after_errors = per_sample_errors(full_result.x, pairs=pairs, fk=fk, fit_delta_q=True)

    write_outputs(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        args=args,
        pairs=pairs,
        offsets_raw=offsets_raw,
        ext_only_seed_name=seed_name,
        ext_result=ext_result,
        full_result=full_result,
        before_errors=before_errors,
        after_errors=after_errors,
    )


if __name__ == "__main__":
    main()
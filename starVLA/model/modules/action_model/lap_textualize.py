# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""LAP-style action textualization.

Convert a continuous action chunk into a LAP-style natural-language
description (arxiv 2602.10556), e.g.::

    "move forward 10 cm, tilt left 5 degrees, rotate clockwise 30 degrees, close gripper"

Format follows LAP Section 3.2:
  - Translation: integer cm per axis ("move {direction} {N} cm")
  - Rotation: Euler XYZ decomposition
      roll  (x): "tilt left/right {N} degrees"
      pitch (y): "tilt forward/backward {N} degrees"
      yaw   (z): "rotate counterclockwise/clockwise {N} degrees"
  - Gripper: last-step state ("open/close gripper", always emitted)

Coordinate convention (+x forward, +y left, +z up, right-hand rule):
  - roll  > 0 → tilt left,          roll  < 0 → tilt right
  - pitch > 0 → tilt forward,       pitch < 0 → tilt backward
  - yaw   > 0 → counterclockwise,   yaw   < 0 → clockwise

This text is used purely as a cross-entropy supervision signal so the VLM
learns to *speak* low-level actions in language aligned with its pre-training
distribution. It is NOT decoded back into actions here -- continuous control
is expected from a separate action expert (AE) added later.
"""

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

# Thresholds: components below these are treated as noise and omitted.
# _TRANS_THRESHOLD >= 0.5 prevents "N cm" where N rounds to 0.
_TRANS_THRESHOLD = 0.5   # nominal cm
_ROT_THRESHOLD_DEG = 2.0  # degrees
_GRIPPER_CLOSE_THRESHOLD = 0.5

_NUMBER_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"


def _net_translation(action: np.ndarray) -> np.ndarray:
    """Net per-axis displacement over the chunk: [dx, dy, dz]."""
    return action[:, :3].sum(axis=0)


def _net_euler_deg(action: np.ndarray):
    """Net rotation as (roll, pitch, yaw) degrees via Rotation composition.

    Composes per-step axis-angle vectors and converts the result to intrinsic
    XYZ Euler angles. Using composition rather than component-wise summation
    correctly handles the non-commutativity of 3-D rotations.
    """
    net = R.identity()
    for rotvec in action[:, 3:6]:
        net = R.from_rotvec(rotvec) * net
    roll, pitch, yaw = net.as_euler("xyz", degrees=True)
    return float(roll), float(pitch), float(yaw)


def textualize_action(action) -> str:
    """Convert an action chunk ``[T, 7]`` to a LAP-style language description.

    Args:
        action: array-like of shape ``[T, 7]`` =
                [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z, gripper].

    Returns:
        A LAP-style description string, e.g.
        ``"move forward 10 cm, tilt left 5 degrees, rotate clockwise 30 degrees, close gripper"``.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"expected action of shape [T, 7], got {action.shape}")

    parts = []

    # --- Translation (integer cm, per axis) ---
    dx, dy, dz = _net_translation(action)
    if abs(dx) > _TRANS_THRESHOLD:
        parts.append(f"move {'forward' if dx > 0 else 'backward'} {round(abs(dx))} cm")
    if abs(dy) > _TRANS_THRESHOLD:
        parts.append(f"move {'left' if dy > 0 else 'right'} {round(abs(dy))} cm")
    if abs(dz) > _TRANS_THRESHOLD:
        parts.append(f"move {'up' if dz > 0 else 'down'} {round(abs(dz))} cm")

    # --- Rotation (Euler XYZ, integer degrees) ---
    roll, pitch, yaw = _net_euler_deg(action)
    if abs(roll) > _ROT_THRESHOLD_DEG:
        parts.append(f"tilt {'left' if roll > 0 else 'right'} {round(abs(roll))} degrees")
    if abs(pitch) > _ROT_THRESHOLD_DEG:
        parts.append(f"tilt {'forward' if pitch > 0 else 'backward'} {round(abs(pitch))} degrees")
    if abs(yaw) > _ROT_THRESHOLD_DEG:
        parts.append(f"rotate {'counterclockwise' if yaw > 0 else 'clockwise'} {round(abs(yaw))} degrees")

    # --- Gripper (always emitted) ---
    parts.append("close gripper" if action[-1, 6] > _GRIPPER_CLOSE_THRESHOLD else "open gripper")

    return ", ".join(parts)


def textualize_action_batch(actions) -> list:
    """Apply :func:`textualize_action` to a batch ``[B][T, 7]`` -> ``List[str]``."""
    return [textualize_action(a) for a in actions]


def language_action_to_interpolated_action(language_action: str, horizon: int, action_dim: int = 7) -> np.ndarray:
    """Decode LAP text into a uniformly interpolated proxy action chunk.

    This is a coarse eval helper for LAP text-only models. It mirrors the
    phrases emitted by :func:`textualize_action`; it is not a general natural
    language action parser.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if action_dim < 7:
        raise ValueError(f"action_dim must be at least 7, got {action_dim}")

    action = np.zeros((horizon, action_dim), dtype=np.float32)
    text = language_action.lower()

    translation = np.zeros(3, dtype=np.float32)
    rotation_deg = np.zeros(3, dtype=np.float32)

    move_pattern = rf"\bmove\s+(forward|backward|left|right|up|down)\s+{_NUMBER_PATTERN}\s+cm\b"
    for direction, value in re.findall(move_pattern, text):
        amount = float(value)
        if direction == "forward":
            translation[0] += amount
        elif direction == "backward":
            translation[0] -= amount
        elif direction == "left":
            translation[1] += amount
        elif direction == "right":
            translation[1] -= amount
        elif direction == "up":
            translation[2] += amount
        elif direction == "down":
            translation[2] -= amount

    tilt_pattern = rf"\btilt\s+(left|right|forward|backward)\s+{_NUMBER_PATTERN}\s+degrees\b"
    for direction, value in re.findall(tilt_pattern, text):
        amount = float(value)
        if direction == "left":
            rotation_deg[0] += amount
        elif direction == "right":
            rotation_deg[0] -= amount
        elif direction == "forward":
            rotation_deg[1] += amount
        elif direction == "backward":
            rotation_deg[1] -= amount

    rotate_pattern = rf"\brotate\s+(counterclockwise|clockwise)\s+{_NUMBER_PATTERN}\s+degrees\b"
    for direction, value in re.findall(rotate_pattern, text):
        amount = float(value)
        rotation_deg[2] += amount if direction == "counterclockwise" else -amount

    action[:, :3] = translation / horizon
    action[:, 3:6] = np.deg2rad(rotation_deg) / horizon

    if re.search(r"\bclose\s+gripper\b", text):
        action[:, 6] = 1.0
    elif re.search(r"\bopen\s+gripper\b", text):
        action[:, 6] = 0.0

    return action


def language_actions_to_interpolated_actions(
    language_actions: list[str],
    horizon: int,
    action_dim: int = 7,
) -> np.ndarray:
    """Apply :func:`language_action_to_interpolated_action` to a batch of LAP strings."""
    return np.stack(
        [
            language_action_to_interpolated_action(text, horizon=horizon, action_dim=action_dim)
            for text in language_actions
        ],
        axis=0,
    )


if __name__ == "__main__":
    # Self-test: deterministic, no GPU / dataset required.
    rng = np.random.default_rng(0)

    # 1. all-zero → only gripper state emitted
    z = np.zeros((16, 7), dtype=np.float32)
    out = textualize_action(z)
    print("zero      :", out)
    assert out == "open gripper", out

    # 2. pure forward translation + close gripper → integer cm
    a = np.zeros((16, 7), dtype=np.float32)
    a[:, 0] = 0.625  # net dx = 10.0
    a[:, 6] = 1.0
    out = textualize_action(a)
    print("forward   :", out)
    assert "move forward 10 cm" in out, out
    assert "close gripper" in out, out

    # 3. pure yaw (z-axis) rotation → counterclockwise (positive yaw)
    a = np.zeros((16, 7), dtype=np.float32)
    a[:, 5] = np.radians(5.0)  # 5 deg/step about z → ~80 deg net yaw
    out = textualize_action(a)
    print("yaw+      :", out)
    assert "rotate counterclockwise" in out, out

    # 4. pure roll (x-axis) rotation → tilt left (positive roll)
    a = np.zeros((16, 7), dtype=np.float32)
    a[:, 3] = np.radians(3.0)  # 3 deg/step about x → 48 deg net roll
    out = textualize_action(a)
    print("roll+     :", out)
    assert "tilt left" in out, out

    # 5. determinism
    a = rng.standard_normal((16, 7)).astype(np.float32) * 0.1
    assert textualize_action(a) == textualize_action(a)

    print("All self-tests passed.")

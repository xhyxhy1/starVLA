# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LAP textualize 抽样预览脚本（一次性验证工具，非框架代码）。

目的：在实现 textualize_action() 之前，用真实 LIBERO 数据抽样，
对比多种 "动作 -> 文本" 方案的实际输出，确定最终格式与量级。

用法：
    python examples/LIBERO/train_files/preview_textualize.py \
        --suite libero_goal --episodes 3 --chunk 16
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DATA_ROOT = "/home/xhy/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA"
SUITE_DIRMAP = {
    "libero_goal": "datasets--IPEC-COMMUNITY--libero_goal_no_noops_1.0.0_lerobot",
    "libero_object": "datasets--IPEC-COMMUNITY--libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial": "datasets--IPEC-COMMUNITY--libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10": "datasets--IPEC-COMMUNITY--libero_10_no_noops_1.0.0_lerobot",
}


def find_snapshot(suite: str) -> str:
    base = os.path.join(DATA_ROOT, SUITE_DIRMAP[suite], "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    if not snaps:
        raise FileNotFoundError(f"未找到 {suite} 的 snapshot：{base}")
    return snaps[0]


def load_action_matrix(parquet_path: str) -> np.ndarray:
    """读取单个 episode 的 action，返回 [T, 7] float32。"""
    df = pd.read_parquet(parquet_path)
    actions = np.stack(df["action"].to_numpy())  # 每行是 np.array(7,)
    return actions.astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# 三种候选 textualize 方案
# ────────────────────────────────────────────────────────────────────
def scheme_a_numeric_net(chunk: np.ndarray) -> str:
    """方案 A：归一化空间净位移，纯数字（最稳，解析 100% 可靠）。"""
    net = chunk[:, :6].sum(axis=0)
    grip = "close" if chunk[-1, 6] > 0.5 else "open"
    return (
        f"dx:{net[0]:+.3f} dy:{net[1]:+.3f} dz:{net[2]:+.3f} "
        f"ax:{net[3]:+.3f} ay:{net[4]:+.3f} az:{net[5]:+.3f} grip:{grip}"
    )


def scheme_b_perstep(chunk: np.ndarray, max_steps: int = 4) -> str:
    """方案 B：逐步数字（前 max_steps 步），保留时序细节。"""
    lines = []
    for i, a in enumerate(chunk[:max_steps]):
        lines.append(
            f"t{i}:[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},"
            f"{a[3]:+.2f},{a[4]:+.2f},{a[5]:+.2f},{a[6]:.0f}]"
        )
    return " ".join(lines)


def net_rotation_deg(chunk: np.ndarray, method: str = "sum") -> float:
    """计算 chunk 的净旋转角(度)。

    method="sum"    : 逐分量求和后取模长 —— 数学错误（旋转不可交换/不可加）。
    method="compose": 逐步 axis-angle 转旋转矩阵连乘 —— 数学正确的净旋转。
    """
    if method == "sum":
        return float(np.degrees(np.linalg.norm(chunk[:, 3:6].sum(axis=0))))
    from scipy.spatial.transform import Rotation as R

    net = R.identity()
    for rv in chunk[:, 3:6]:
        net = R.from_rotvec(rv) * net
    return float(np.degrees(net.magnitude()))


def scheme_lap(chunk: np.ndarray) -> str:
    """LAP 论文形式：<verb> <direction> <magnitude> <unit>。不要求可逆。

    - 平移：净位移 -> 名义 cm（scale=1），方向词基于 robosuite 惯例。
    - 旋转：axis-angle 净向量的模长 = 旋转角(rad) -> degrees（不标轴）。
    - gripper：末步状态 -> open/close。
    现阶段只作 CE 监督信号，推理端动作由后续 AE 生成，故无需解析回动作。
    """
    net = chunk[:, :6].sum(axis=0)
    parts = []
    if abs(net[0]) > 0.3:
        parts.append(f"move {'forward' if net[0] > 0 else 'backward'} {abs(net[0]):.1f} cm")
    if abs(net[1]) > 0.3:
        parts.append(f"move {'left' if net[1] > 0 else 'right'} {abs(net[1]):.1f} cm")
    if abs(net[2]) > 0.3:
        parts.append(f"move {'up' if net[2] > 0 else 'down'} {abs(net[2]):.1f} cm")
    rot_deg = np.degrees(np.linalg.norm(net[3:6]))  # axis-angle 模长即旋转角
    if rot_deg > 2.0:
        parts.append(f"rotate {rot_deg:.1f} degrees")
    parts.append("close gripper" if chunk[-1, 6] > 0.5 else "open gripper")
    return ", ".join(parts) if parts else "hold position"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal", choices=list(SUITE_DIRMAP))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    snap = find_snapshot(args.suite)
    parquets = sorted(glob.glob(os.path.join(snap, "data", "**", "*.parquet"), recursive=True))
    print(f"[suite={args.suite}] 共 {len(parquets)} 个 episode, chunk={args.chunk} (fps=20 -> {args.chunk/20:.2f}s/chunk)\n")

    # 全局动作统计（用所抽 episode 估计量级）
    all_actions = []
    for pq in parquets[: args.episodes]:
        ep = int(os.path.basename(pq).split("_")[-1].split(".")[0])
        acts = load_action_matrix(pq)
        all_actions.append(acts)
        print(f"================ Episode {ep}  (T={len(acts)}) ================")

        # 取 3 个代表性 chunk：开头、中段、gripper 翻转附近
        grip = acts[:, 6]
        flip_idx = np.where(np.abs(np.diff(grip)) > 0.5)[0]
        sample_starts = [0, len(acts) // 2]
        if len(flip_idx) > 0:
            sample_starts.append(max(0, int(flip_idx[0]) - args.chunk // 2))

        for s in sample_starts:
            chunk = acts[s : s + args.chunk]
            if len(chunk) < 2:
                continue
            print(f"\n  --- chunk[{s}:{s+len(chunk)}] ---")
            print(f"  [原始净位移] {chunk[:, :6].sum(axis=0).round(3).tolist()}  grip {chunk[0,6]:.0f}->{chunk[-1,6]:.0f}")
            print(f"  [A 数字净位移] {scheme_a_numeric_net(chunk)}")
            print(f"  [B 逐步数字  ] {scheme_b_perstep(chunk)}")
            print(f"  [LAP 论文形式 ] {scheme_lap(chunk)}")
            print(f"  [旋转角对比  ] 逐分量求和(当前/错)={net_rotation_deg(chunk,'sum'):.1f}deg  "
                  f"连乘(正确)={net_rotation_deg(chunk,'compose'):.1f}deg")
        print()

    # 量级汇总
    cat = np.concatenate(all_actions, axis=0)
    print("================ 动作量级汇总（抽样 episode）================")
    names = ["x", "y", "z", "ax_angle1", "ax_angle2", "ax_angle3", "gripper"]
    print(f"{'dim':<12}{'min':>9}{'max':>9}{'mean':>9}{'abs_mean':>9}")
    for i, n in enumerate(names):
        col = cat[:, i]
        print(f"{n:<12}{col.min():>9.3f}{col.max():>9.3f}{col.mean():>9.3f}{np.abs(col).mean():>9.3f}")
    print(f"\n单步平移幅值均值: {np.abs(cat[:, :3]).mean():.4f}")
    print(f"单步旋转幅值均值: {np.abs(cat[:, 3:6]).mean():.4f}")
    print(f"{args.chunk}步累计平移幅值(估): {np.abs(cat[:, :3]).mean()*args.chunk:.4f}")


if __name__ == "__main__":
    main()

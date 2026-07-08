#!/usr/bin/env python3
"""Replace HF pointer files with blob content for libero_goal, object, spatial."""

import os
from pathlib import Path

ROOT = Path("/home/xhy/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA")
SUBSETS = {
    "libero_goal_no_noops_1.0.0_lerobot": "222cf888ed360fad0a5f983748c1cc40743d43e7",
    "libero_object_no_noops_1.0.0_lerobot": "15657dac2ad1c01b4e94bf54ab0493b46a8d63f9",
    "libero_spatial_no_noops_1.0.0_lerobot": "bf14d6258218d12c2e3c1a3b9922e163cdf6455d",
}


def resolve(base: Path, ref: str) -> Path:
    return Path(os.path.normpath(base / ref))


def fix_meta(snap: Path, hf_root: Path):
    meta = snap / "meta"
    for name in ["info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]:
        p = meta / name
        if not p.is_file():
            continue
        with open(p) as f:
            ref = f.read().strip()
        if not ref.startswith("."):
            continue
        blob = resolve(p.parent, ref)
        if blob.is_file():
            p.write_bytes(blob.read_bytes())
            print(f"  meta/{name} <- blob")


def fix_parquets(snap: Path):
    data = snap / "data" / "chunk-000"
    if not data.is_dir():
        return
    n = 0
    for f in data.glob("episode_*.parquet"):
        with open(f) as fp:
            ref = fp.read().strip()
        if not ref.startswith("."):
            continue
        blob = resolve(f.parent, ref)
        if blob.is_file():
            f.write_bytes(blob.read_bytes())
            n += 1
    print(f"  parquets: {n}")


def fix_videos(snap: Path):
    vid = snap / "videos" / "chunk-000"
    if not vid.is_dir():
        return
    n = 0
    for sub in vid.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*.mp4"):
            try:
                ref = f.read_text().strip()
            except Exception:
                continue
            if not ref.startswith("."):
                continue
            blob = resolve(f.parent, ref)
            if blob.is_file():
                f.write_bytes(blob.read_bytes())
                n += 1
    print(f"  videos: {n}")


def main():
    for short_name, snap_hash in SUBSETS.items():
        hf_name = f"datasets--IPEC-COMMUNITY--{short_name}"
        hf_root = ROOT / hf_name
        snap = hf_root / "snapshots" / snap_hash
        if not snap.is_dir():
            print(f"skip {short_name}: snapshot not found")
            continue
        print(short_name)
        fix_meta(snap, hf_root)
        fix_parquets(snap)
        fix_videos(snap)

        # symlink
        link = ROOT / short_name
        if link.exists():
            if link.is_symlink():
                print(f"  symlink exists -> {link.resolve()}")
            else:
                print(f"  skip link: {link} exists and is not symlink")
        else:
            link.symlink_to(snap)
            print(f"  symlink {short_name} -> {snap}")


if __name__ == "__main__":
    main()

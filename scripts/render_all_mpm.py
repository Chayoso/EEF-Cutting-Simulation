"""Batch driver: render mpm.mp4 for every collected episode in a dataset tree.

Dataset layout (CulinaryCut Drive dump):
    <dataset-root>/<fruit>/auto_<seed>/mpm_render.npz   (knife tip, MPM frame)
                                       trajectory.json   (variation.object)
                                       mpm.mp4           (old/broken reference)

For each episode it calls render_mpm_episode_portable.py with the matching
<config-dir>/<fruit>.yaml and writes the new video next to the npz.

Parallelism: shard across N server processes with --shard/--of (each process
handles episodes where global_index % of == shard).

Example (single process, all fruits):
    python scripts/render_all_mpm.py \
        --dataset-root /data/culinary_traj \
        --repo /data/CPIC \
        --config-dir configs/fruits \
        --damage-v-hat 0.0007 --out-name mpm_fixed.mp4

Example (8-way parallel on the server):
    for s in $(seq 0 7); do
      python scripts/render_all_mpm.py --dataset-root /data/culinary_traj \
        --repo /data/CPIC --config-dir configs/fruits \
        --damage-v-hat 0.0007 --out-name mpm_fixed.mp4 --of 8 --shard $s &
    done; wait
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

RENDER = Path(__file__).resolve().parent / "render_mpm_episode_portable.py"


def fruit_of(ep_dir: Path) -> str:
    """Fruit name from trajectory.json variation, else the grandparent dir name."""
    tj = ep_dir / "trajectory.json"
    if tj.exists():
        try:
            d = json.loads(tj.read_text(encoding="utf-8"))
            obj = d["episodes"][0]["reset_kwargs"]["options"]["variation"]["object"]
            if obj:
                return str(obj)
        except Exception:
            pass
    return ep_dir.parent.name  # <root>/<fruit>/auto_<seed>


def find_episodes(root: Path):
    eps = []
    for npz in sorted(root.glob("*/auto_*/mpm_render.npz")):
        eps.append(npz.parent)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="CPIC repo root")
    ap.add_argument("--config-dir", type=str, default="configs/fruits")
    ap.add_argument("--out-name", type=str, default="mpm_fixed.mp4")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--damage-v-hat", type=float, default=None)
    ap.add_argument("--knife-speed", type=float, default=None)
    ap.add_argument("--extra-descent", type=float, default=0.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    repo = args.repo.resolve()
    cfg_dir = (repo / args.config_dir)
    eps = find_episodes(args.dataset_root.resolve())
    eps = [e for i, e in enumerate(eps) if i % args.of == args.shard]
    if args.limit > 0:
        eps = eps[: args.limit]

    print(f"[batch] shard {args.shard}/{args.of}  episodes={len(eps)}  out={args.out_name}", flush=True)
    n_ok = n_err = n_skip = 0
    for i, ep in enumerate(eps):
        fruit = fruit_of(ep)
        cfg = cfg_dir / f"{fruit}.yaml"
        out = ep / args.out_name
        if out.exists() and not args.overwrite:
            n_skip += 1
            print(f"  [{i+1}/{len(eps)}] {fruit}/{ep.name}: skip (exists)", flush=True)
            continue
        if not cfg.exists():
            n_err += 1
            print(f"  [{i+1}/{len(eps)}] {fruit}/{ep.name}: ERR no config {cfg}", flush=True)
            continue
        cmd = [args.python, str(RENDER),
               "--npz", str(ep / "mpm_render.npz"),
               "--config", str(cfg),
               "--repo", str(repo),
               "--out", str(out),
               "--fps", str(args.fps), "--res", str(args.res),
               "--extra-descent", f"{args.extra_descent:.4f}"]
        if args.damage_v_hat is not None:
            cmd += ["--damage-v-hat", f"{args.damage_v_hat}"]
        if args.knife_speed is not None:
            cmd += ["--knife-speed", f"{args.knife_speed}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            if r.returncode == 0 and out.exists():
                n_ok += 1
                print(f"  [{i+1}/{len(eps)}] {fruit}/{ep.name}: ok -> {out.name}", flush=True)
            else:
                n_err += 1
                print(f"  [{i+1}/{len(eps)}] {fruit}/{ep.name}: ERR rc={r.returncode} {r.stderr[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            n_err += 1
            print(f"  [{i+1}/{len(eps)}] {fruit}/{ep.name}: ERR timeout", flush=True)
    print(f"[batch] done ok={n_ok} err={n_err} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()

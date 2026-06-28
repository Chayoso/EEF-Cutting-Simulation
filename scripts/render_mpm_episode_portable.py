"""Portable replay-render of an MPM episode -> mpm.mp4 (local CPIC, Windows/Linux).

Based on mani_skill/dataset_converters/mpm/render_mpm_episode.py but:
  - no hardcoded /data paths; repo root via --repo (default: cwd)
  - diagnostics (--diag): seeded-particle AABB vs knife-trajectory AABB
  - optional --frames-dir to keep PNGs for inspection
  - optional --anchor to re-center knife trajectory onto the actually-seeded fruit

Usage:
  python render_mpm_local.py --npz <mpm_render.npz> --config <fruit.yaml> \
      --repo <CPIC root> --out mpm.mp4 --diag --frames-dir <dir>
"""
from __future__ import annotations
import argparse, os, sys, subprocess, tempfile, shutil
import numpy as np, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--mpm-substeps-cap", type=int, default=100)
    ap.add_argument("--extra-descent", type=float, default=0.0)
    ap.add_argument("--damage-v-hat", type=float, default=None,
                    help="override damage_v_hat so the cut triggers at replay speed "
                         "(threshold_mps = dx*v_hat/dt). e.g. 0.0007 ~= 0.15 m/s")
    ap.add_argument("--knife-speed", type=float, default=None,
                    help="override knife current/base speed reported to the damage model")
    ap.add_argument("--anchor", action="store_true",
                    help="shift knife xz so it lands on the actually-seeded fruit centroid")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)

    data = np.load(args.npz, allow_pickle=False)
    knife_traj = data["knife"].astype(np.float32)  # (T,3) MPM frame
    T = knife_traj.shape[0]
    if args.max_frames > 0:
        T = min(T, args.max_frames)
        knife_traj = knife_traj[:T]
    print(f"[render] knife T={T}", flush=True)

    import taichi as ti
    try:
        ti.init(arch=ti.cuda, random_seed=1)
    except Exception:
        ti.init(arch=ti.vulkan)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from sdf_utils.mesh_sdf import mesh_to_sdf
    import trimesh

    def _pack(block, with_blade):
        transform = block.get("initial_transform", {})
        kw = {}
        if with_blade:
            kw["knife_blade"] = block.get("blade", {"axis": "Y", "fraction": 0.5})
        return mesh_to_sdf(block["mesh_path"], transform, int(block["sdf_voxel"]), **kw)

    def _box_pack(block, padding=0.01):
        """Analytic box SDF from the mesh AABB. The cutting board is a flat slab;
        trimesh.proximity.signed_distance crashes natively on the non-watertight
        board mesh, so we build a robust analytic slab instead (portable)."""
        m = trimesh.load(block["mesh_path"], force="mesh")
        bmin = np.asarray(m.bounds[0], np.float32) - padding
        bmax = np.asarray(m.bounds[1], np.float32) + padding
        size = bmax - bmin
        maxlen = float(np.max(size))
        vsize = maxlen / float(int(block["sdf_voxel"]))
        Nx = max(int(np.floor(size[0] / vsize)) + 1, 2)
        Ny = max(int(np.floor(size[1] / vsize)) + 1, 2)
        Nz = max(int(np.floor(size[2] / vsize)) + 1, 2)
        xs = bmin[0] + vsize * np.arange(Nx, dtype=np.float32)
        ys = bmin[1] + vsize * np.arange(Ny, dtype=np.float32)
        zs = bmin[2] + vsize * np.arange(Nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")  # (Ny,Nx,Nz)
        pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
        c = (np.asarray(m.bounds[0], np.float32) + np.asarray(m.bounds[1], np.float32)) / 2.0
        h = (np.asarray(m.bounds[1], np.float32) - np.asarray(m.bounds[0], np.float32)) / 2.0
        q = np.abs(pts - c) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        d = (outside + inside).astype(np.float32)  # inside<0
        sdf = d.reshape(Ny, Nx, Nz).transpose(2, 0, 1).astype(np.float32)  # [Nz,Ny,Nx]
        return {
            "sdf": sdf,
            "origin": bmin.astype(np.float32),
            "voxel_size": np.float32(vsize),
            "grid_shape": (int(Nz), int(Ny), int(Nx)),
        }

    cutting_pack = _pack(cfg["cutting_mesh"], False)
    knife_pack = _pack(cfg["knife"], True)
    board_pack = _box_pack(cfg["board"]) if "board" in cfg else None
    cfg["cutting_mesh_pack"] = cutting_pack

    from mpmcore.sim import MPMCuttingSim
    sim = MPMCuttingSim(cfg, cutting_pack, knife_pack, board_pack=board_pack, viewer=False)
    sim.viewer_enabled = True
    sim.viewer_camera_mode = "manual"
    sim.viewer_lock_on_run = True
    sim.window = ti.ui.Window("MPM", res=(args.res, args.res), vsync=False, show_window=False)
    sim.canvas = sim.window.get_canvas()
    sim.scene = sim.window.get_scene()
    sim.camera = ti.ui.Camera()

    top = None
    if sim.board is not None:
        try:
            top = float(sim.board_top_y[None])
        except Exception:
            top = None
    sim.seed_particles_from_mesh(cutting_pack, cfg["cutting_mesh"], name="cutting_mesh", support_top_y=top)

    # Optional: make the cut trigger at the (slow) replay knife speed.
    if args.damage_v_hat is not None:
        sim.damage_v_hat_f[None] = float(args.damage_v_hat)
        sim.damage_v_threshold_f[None] = float(sim.dx_s[None]) * float(args.damage_v_hat) / max(1e-6, float(sim.dt))
        print(f"[cut] damage_v_hat={args.damage_v_hat} -> threshold={float(sim.damage_v_threshold_f[None]):.4f} m/s", flush=True)
    if args.knife_speed is not None:
        sim.knife.current_speed[None] = float(args.knife_speed)
        sim.knife.base_speed = float(args.knife_speed)
        print(f"[cut] knife reported speed -> {args.knife_speed} m/s", flush=True)

    n = int(sim.pcount[None])
    P = sim.particles.x.to_numpy()[:n]
    pmin, pmax = P.min(0), P.max(0)
    pmid = (pmin + pmax) / 2.0
    kmin, kmax = knife_traj.min(0), knife_traj.max(0)
    if args.diag:
        print(f"[diag] particles={n}")
        print(f"[diag] particle AABB  min={pmin}  max={pmax}  mid={pmid}")
        print(f"[diag] knife traj AABB min={kmin}  max={kmax}")
        print(f"[diag] board_top_y={top}  knife_yfoot={sim._knife_yfoot}")
        kb = sim._knife_base_xyz
        print(f"[diag] knife_base_xyz AABB min={kb.min(0)} max={kb.max(0)} (x fixed, only y/z_off driven)")
        print(f"[diag] deepest knife tip y={kmin[1]:.4f} vs particle y range [{pmin[1]:.4f},{pmax[1]:.4f}]")

    # Optional anchor: shift knife xz to land on the seeded fruit centroid (xz).
    anchor_dxz = np.zeros(2, np.float32)
    if args.anchor:
        kmid_xz = np.array([(kmin[0] + kmax[0]) / 2, (kmin[2] + kmax[2]) / 2], np.float32)
        anchor_dxz = np.array([pmid[0] - kmid_xz[0], pmid[2] - kmid_xz[2]], np.float32)
        print(f"[anchor] shifting knife xz by {anchor_dxz}")

    def _ext_update(dt):
        sim.knife.sim_time = float(getattr(sim.knife, "sim_time", 0.0)) + float(dt)
    sim.knife.update = _ext_update

    cen = pmid
    sim.camera.position(float(cen[0] + 0.28), float(cen[1] + 0.20), float(cen[2] + 0.32))
    sim.camera.lookat(float(cen[0]), float(cen[1]), float(cen[2]))
    sim.camera.up(0, 1, 0)
    sim.camera.fov(35.0)

    blend_hi, blend_lo = 0.20, 0.11

    def drive(tip):
        y = float(tip[1])
        t = (blend_hi - y) / max(1e-6, (blend_hi - blend_lo))
        t = max(0.0, min(1.0, t))
        s = t * t * (3.0 - 2.0 * t)
        extra = float(args.extra_descent) * s
        sim.knife.y[None] = y - sim._knife_yfoot - extra
        if hasattr(sim.knife, "z_off"):
            sim.knife.z_off[None] = float(tip[2] + anchor_dxz[1])

    from render_utils.renderer import MPMRenderer
    renderer = MPMRenderer(sim)

    dt_frame = float(sim.dt) * int(sim.substeps)
    n_sub = max(1, int(round(args.control_dt / dt_frame)))
    if args.mpm_substeps_cap > 0:
        n_sub = min(n_sub, args.mpm_substeps_cap)
    print(f"[render] substeps_per_control={n_sub}", flush=True)

    _t = 0.0
    keep = args.frames_dir is not None
    tmp = args.frames_dir or tempfile.mkdtemp(prefix="mpm_render_")
    os.makedirs(tmp, exist_ok=True)
    max_damage = 0.0
    try:
        for t in range(T):
            drive(knife_traj[t])
            for _ in range(n_sub):
                sim.step(getattr(sim, "sim_time", _t))
                _t += dt_frame
            try:
                sim.scene.set_camera(sim.camera)
                renderer._setup_scene_lighting()
                renderer._render_particles()
                renderer._render_knife_proxy()
                renderer._render_eef_position()
                if getattr(sim, "show_grid", False):
                    renderer._render_grid_visualization()
                sim.canvas.scene(sim.scene)
            except Exception as e:
                print(f"[render] frame {t} draw err: {e}", flush=True)
                continue
            sim.window.save_image(f"{tmp}/f{t:05d}.png")
            if args.diag and (t % 40 == 0 or t == T - 1):
                D = sim.particles.D.to_numpy()[:n]
                md = float(D.max()) if D.size else 0.0
                max_damage = max(max_damage, md)
                print(f"[diag] frame {t}: max_damage={md:.4f} knife.y={float(sim.knife.y[None]):.4f} z_off={float(sim.knife.z_off[None]):.4f}", flush=True)

        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"
        cmd = [ffmpeg, "-y", "-framerate", str(args.fps), "-i", f"{tmp}/f%05d.png",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", args.out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:], file=sys.stderr); sys.exit(3)
        print(f"[render] wrote {args.out}  (max_damage seen={max_damage:.4f})", flush=True)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()


"""
SDF‑based collider primitives for the cutting demo.
- BaseCollider: sampling, normals, optional mask, and basic motion hooks
- KnifeCollider: animated knife with blade mask and contact classification
- BoardCollider: static chopping board
"""


from __future__ import annotations
import numpy as np
import taichi as ti

@ti.data_oriented
class BaseCollider:
    """
    Base interface for SDF-based colliders.
    - Provides SDF samples and normals in world space.
    - Provides local velocity of the collider at a world-space point (for friction/impact).
    - Updates internal animation parameters each step (default: static).
    - Exposes approximate world-space Y-range of the solid region.
    """
    def __init__(self, sdf_grid: np.ndarray, origin, voxel_size: float, friction: float=0.3, restitution: float=0.0, mask_grid: np.ndarray=None):
        # Numpy handles (CPU-side)
        self._sdf_np = np.asarray(sdf_grid, np.float32)
        self._origin_py = np.asarray(origin, np.float32).copy()
        self._voxel_py  = float(voxel_size)

        # Taichi fields (GPU-side)
        self.sdf = ti.field(dtype=ti.f32, shape=self._sdf_np.shape)
        self.sdf.from_numpy(self._sdf_np)
        self.origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.voxel  = ti.field(dtype=ti.f32, shape=())
        self.origin[None] = ti.Vector(self._origin_py.tolist())
        self.voxel[None]  = self._voxel_py
        self.friction_f   = ti.field(dtype=ti.f32, shape=()); self.friction_f[None] = float(friction)
        self.restitution_f= ti.field(dtype=ti.f32, shape=()); self.restitution_f[None] = float(restitution)

        # Optional mask grid (e.g., knife blade mask)
        self._has_mask = int(mask_grid is not None)
        if self._has_mask:
            self._mask_np = np.asarray(mask_grid, np.float32)
            if self._mask_np.shape != self._sdf_np.shape:
                raise ValueError("mask_grid shape must match sdf_grid shape")
            self.mask = ti.field(dtype=ti.f32, shape=self._mask_np.shape)
            self.mask.from_numpy(self._mask_np)
        else:
            self._mask_np = None
            self.mask = None

        # Precompute solid Y-index range for CPU-side logic (Nz, Ny, Nx) -> axis=1 is Y
        solid = self._sdf_np < 0.0
        if np.any(solid):
            ys = np.where(solid)[1]
            self._yidx_min = int(ys.min())
            self._yidx_max = int(ys.max())
        else:
            self._yidx_min = 0
            self._yidx_max = -1  # empty

    # World-space Y of a given Y-index (cell center)
    def _y_index_to_world(self, y_idx: int) -> float:
        return float(self.origin[None][1] + (y_idx + 0.5) * self._voxel_py)

    @property
    def solid_ymin_world(self) -> float:
        if self._yidx_max < self._yidx_min:  # empty
            return float(self.origin[None][1])
        return self._y_index_to_world(self._yidx_min)

    @property
    def solid_ymax_world(self) -> float:
        if self._yidx_max < self._yidx_min:
            return float(self.origin[None][1])
        return self._y_index_to_world(self._yidx_max)

    # --------- sampling ---------
    @ti.func
    def _sample_grid(self, grid, p, y_anim, z_off) -> ti.f32:
        """Trilinear sample 'grid' at world point p using dynamic Y/Z offsets."""
        o = self.origin[None]
        v = self.voxel[None]
        # Optimize: precompute offset vector
        offset = ti.Vector([0.0, 0.0, z_off])
        o_dyn = ti.Vector([o[0], y_anim, o[2]])
        q = (p - o_dyn - offset) / v
        xi, yi, zi = q[0], q[1], q[2]
        x0 = ti.cast(ti.floor(xi), ti.i32); x1 = x0+1
        y0 = ti.cast(ti.floor(yi), ti.i32); y1 = y0+1
        z0 = ti.cast(ti.floor(zi), ti.i32); z1 = z0+1
        Nx = ti.static(grid.shape[2]); Ny = ti.static(grid.shape[1]); Nz = ti.static(grid.shape[0])
        x0 = ti.max(0, ti.min(Nx-1, x0)); x1 = ti.max(0, ti.min(Nx-1, x1))
        y0 = ti.max(0, ti.min(Ny-1, y0)); y1 = ti.max(0, ti.min(Ny-1, y1))
        z0 = ti.max(0, ti.min(Nz-1, z0)); z1 = ti.max(0, ti.min(Nz-1, z1))
        xd = xi - ti.cast(x0, ti.f32); yd = yi - ti.cast(y0, ti.f32); zd = zi - ti.cast(z0, ti.f32)
        # Optimize: precompute interpolation weights
        xd_inv = 1.0 - xd; yd_inv = 1.0 - yd; zd_inv = 1.0 - zd
        c000=grid[z0,y0,x0]; c100=grid[z0,y0,x1]
        c010=grid[z0,y1,x0]; c110=grid[z0,y1,x1]
        c001=grid[z1,y0,x0]; c101=grid[z1,y0,x1]
        c011=grid[z1,y1,x0]; c111=grid[z1,y1,x1]
        c00=c000*xd_inv+c100*xd; c01=c001*xd_inv+c101*xd
        c10=c010*xd_inv+c110*xd; c11=c011*xd_inv+c111*xd
        c0=c00*yd_inv+c10*yd;    c1=c01*yd_inv+c11*yd
        return c0*zd_inv+c1*zd

    @ti.func
    def sample(self, p):
        """SDF sample at world point p (negative = inside). Override in subclasses to add motion."""
        return self._sample_grid(self.sdf, p, self.origin[None][1], 0.0)

    @ti.func
    def sample_mask(self, p):
        """Sample optional mask at world point p (1=blade, 0=not blade)."""
        if ti.static(self.mask == None):
            return 1.0
        # Default collider has no motion; subclasses can override with animation state
        return self._sample_grid(self.mask, p, self.origin[None][1], 0.0)

    @ti.func
    def normal(self, p):
        """SDF gradient-based normal (outward)."""
        h = 0.75 * self.voxel[None]  # Smoother normal for high resolution
        sx = self.sample(p + ti.Vector([ti.f32(h), ti.f32(0.0), ti.f32(0.0)])) - self.sample(p - ti.Vector([ti.f32(h), ti.f32(0.0), ti.f32(0.0)]))
        sy = self.sample(p + ti.Vector([ti.f32(0.0), ti.f32(h), ti.f32(0.0)])) - self.sample(p - ti.Vector([ti.f32(0.0), ti.f32(h), ti.f32(0.0)]))
        sz = self.sample(p + ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(h)])) - self.sample(p - ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(h)]))
        n = ti.Vector([ti.f32(sx), ti.f32(sy), ti.f32(sz)])
        return n / (n.norm() + 1e-8)

    @ti.func
    def velocity_at(self, p):
        """Rigid velocity at world point (default static)."""
        return ti.Vector([ti.f32(0.0), ti.f32(0.0), ti.f32(0.0)])

    def update(self, dt: float):
        """Advance internal animation; default no-op."""
        pass


@ti.data_oriented
class KnifeCollider(BaseCollider):
    """
    Moving knife with vertical Y motion and optional Z offsets.
    Optional blade mask limits cutting to the blade region while handle only collides.
    """
    def __init__(self, sdf_grid, origin, voxel_size, start_y, stop_y,
                 speed, return_speed, z_offset=0.0, friction=0.3,
                 blade_mask_grid=None, mesh_restitution=0.01):
        super().__init__(sdf_grid, origin, voxel_size, friction, 0.0, mask_grid=blade_mask_grid)
        self.y = ti.field(dtype=ti.f32, shape=()); self.y[None] = float(start_y)
        self.start_y = float(start_y); self.stop_y = float(stop_y)
        self.speed = float(speed);     self.return_speed = float(return_speed)
        self.z_off = ti.field(dtype=ti.f32, shape=()); self.z_off[None] = float(z_offset)
        self._down = ti.field(dtype=ti.i32, shape=()); self._down[None] = 1
        self._knife_yidx_min = getattr(self, "_yidx_min", 0)
        
        # Board collision tracking fields
        self.has_board_flag = ti.field(dtype=ti.i32, shape=()); self.has_board_flag[None] = 0
        self._hit_mesh = ti.field(dtype=ti.i32, shape=()); self._hit_mesh[None] = 0
        self._hit_board = ti.field(dtype=ti.i32, shape=()); self._hit_board[None] = 0
        self._board = None
        
        # Mesh restitution for cutting simulation
        self.mesh_restitution_f = ti.field(dtype=ti.f32, shape=()); self.mesh_restitution_f[None] = float(mesh_restitution)
        
        # Smooth speed control (tunable)
        self.base_speed = float(speed)
        self.current_speed = ti.field(dtype=ti.f32, shape=()); self.current_speed[None] = float(speed)
        self.min_speed_f = ti.field(dtype=ti.f32, shape=()); self.min_speed_f[None] = 0.8 * float(speed)  # 25% floor
        # Apply restitution to cut_tau: lower restitution = slower decay (less speed reduction)
        base_tau = 1.0  # [s] base decay time constant (increased for slower speed changes)
        restitution_factor = 1.0 + (1.0 - float(mesh_restitution))  # 1.0 to 2.0 range (inverted)
        self.cut_tau_f = ti.field(dtype=ti.f32, shape=()); self.cut_tau_f[None] = base_tau * restitution_factor
        self.rec_tau_f = ti.field(dtype=ti.f32, shape=()); self.rec_tau_f[None] = 0.6   # [s] recovery time constant
        self.c_norm_f = ti.field(dtype=ti.f32, shape=()); self.c_norm_f[None] = 1.0    # [m/s] normalize contact metric (increased for less sensitivity)
        
        # Per-substep contact metric accumulator (atomic-updated in grid_respond)
        self.contact_accum_f = ti.field(dtype=ti.f32, shape=()); self.contact_accum_f[None] = 0.0
        

    def world_low_y(self) -> float:
        """Lowest voxel center height = y_anim + (ymin_idx+0.5)*voxel"""
        return float(self.y[None]) + (self._knife_yidx_min + 0.5) * float(self._voxel_py)

    def force_up(self):
        """Force the knife to start moving upward from the next update."""
        self._down[None] = 0

    def update(self, dt: float):
        """Ping-pong between start_y and stop_y with smooth speed control."""
        # Ping-pong Y motion (existing)
        if self._down[None] == 1:
            y_new = self.y[None] - dt * self.current_speed[None]
            if y_new <= self.stop_y:
                y_new = self.stop_y; self._down[None] = 0
        else:
            y_new = self.y[None] + dt * self.return_speed
            if y_new >= self.start_y:
                y_new = self.start_y; self._down[None] = 1
        self.y[None] = y_new
        
        # Physical speed control based on cutting resistance
        # Pull out accumulated contact metric (Σ max(0, -vn_rel)) and reset
        c = float(self.contact_accum_f[None]); self.contact_accum_f[None] = 0.0
        c_hat = min(1.0, c / max(1e-4, float(self.c_norm_f[None])))  # normalize to [0,1]
        
        s = float(self.current_speed[None])
        s0 = float(self.base_speed)
        s_min = float(self.min_speed_f[None])
        
        # Physical cutting resistance model - enhanced for stronger speed reduction
        if int(self._down[None]) == 1 and c_hat > 1e-4:
            # Model cutting resistance as a force opposing knife motion
            # F_resistance = k * contact_area * velocity^2 (quadratic drag)
            # This creates more realistic speed reduction under load
            resistance_coeff = 2.0  # Increased resistance coefficient (0.5 → 2.0)
            velocity_factor = (s / s0) ** 1.5  # Moderate velocity dependence (2.0 → 1.5)
            resistance_force = resistance_coeff * c_hat * velocity_factor
            
            # Apply resistance as acceleration: a = F/m (simplified mass = 1)
            # Speed change: dv = -a * dt
            speed_reduction = resistance_force * dt
            s = max(s_min, s - speed_reduction)
        
        # Natural recovery toward base speed (like spring restoring force)
        if int(self._down[None]) == 0 or c_hat < 1e-4:
            # Recovery force proportional to speed difference
            recovery_force = 0.3  # Increased recovery coefficient (0.1 → 0.3)
            speed_difference = s0 - s
            speed_increase = recovery_force * speed_difference * dt
            s = min(s0, s + speed_increase)
        
        # Clamp between [s_min, s0]
        s = max(s_min, min(s, s0))
        self.current_speed[None] = s

    @ti.func
    def sample(self, p):
        """SDF sample with animated Y and Z offsets."""
        return self._sample_grid(self.sdf, p, self.y[None], self.z_off[None])

    @ti.func
    def sample_mask(self, p):
        """Mask sample with animated Y and Z offsets (1=blade, 0=not blade)."""
        if ti.static(self.mask == None):
            return 1.0
        return self._sample_grid(self.mask, p, self.y[None], self.z_off[None])

    @ti.func
    def is_blade(self, p):
        """Return 1 if inside the blade region; 0 otherwise."""
        return 1 if self.sample_mask(p) > 0.5 else 0

    @ti.func
    def normal(self, p):
        h = 0.75 * self.voxel[None]  # Smoother normal for high resolution
        sx = self.sample(p + ti.Vector([h, 0.0, 0.0])) - self.sample(p - ti.Vector([h, 0.0, 0.0]))
        sy = self.sample(p + ti.Vector([0.0, h, 0.0])) - self.sample(p - ti.Vector([0.0, h, 0.0]))
        sz = self.sample(p + ti.Vector([0.0, 0.0, h])) - self.sample(p - ti.Vector([0.0, 0.0, h]))
        n = ti.Vector([sx, sy, sz])
        return n / (n.norm() + 1e-8)

    @ti.func
    def velocity_at(self, p):
        # Knife moves along -Y / +Y (use current_speed for accurate collision response)
        vy = -self.current_speed[None] if self._down[None]==1 else self.return_speed
        return ti.Vector([ti.f32(0.0), ti.f32(vy), ti.f32(0.0)])

    def attach_board(self, board):
        """Attach a BoardCollider instance so the knife can classify board vs mesh collisions itself."""
        self._board = board
        try:
            # Mark presence of board for Taichi-side logic
            self.has_board_flag[None] = 1
        except Exception:
            pass

    @ti.kernel
    def reset_contact_counters(self):
        self._hit_mesh[None] = 0
        self._hit_board[None] = 0

    @ti.func
    def classify_at(self, p) -> ti.i32:
        """Return 1 if knife-vs-mesh, 2 if knife-vs-board, 0 if no collision at point p."""
        dknife = self.sample(p)
        
        # --- board sampling guarded at compile time and runtime ---
        db = 1e9
        eps_b = ti.f32(0.0)
        if ti.static(self._board != None) and (self.has_board_flag[None] == 1):
            db = self._board.sample(p)
            eps_b = ti.f32(0.25) * self._board.voxel[None]
        
        # knife epsilon (always available)
        eps_k = ti.f32(0.25) * self.voxel[None]
        
        result = 0
        knife_in = dknife < eps_k
        board_in = (eps_b > 0.0) and (db < eps_b)
        
        # Prefer knife when both are in contact and knife is closer
        if knife_in and (not board_in or (dknife < db)):
            result = 1
        elif board_in:
            result = 2
        
        return result

    @ti.func
    def grid_respond(self, pos, v):
        """
        Collision response for grid velocity at 'pos'.
        Applies reflection/friction only when approaching (energy safe).
        """
        which = self.classify_at(pos)
        if which == 1:
            # Knife ↔ Mesh (use knife's restitution/friction)
            n = self.normal(pos)
            vknife = self.velocity_at(pos)
            v_rel = v - vknife
            vn_rel = v_rel.dot(n)
            if vn_rel < 0.0:
                # Accumulate contact metric (normal approach speed), blade-only
                if self.is_blade(pos) == 1:
                    ti.atomic_add(self.contact_accum_f[None], -vn_rel)  # unit: m/s
                
                # Reflect normal component with mesh restitution (energy loss)
                vn_rel_new = - self.mesh_restitution_f[None] * vn_rel
                vt_rel = v_rel - vn_rel * n
                # Coulomb friction with slip clamp
                vt_len = vt_rel.norm()
                if vt_len > 1e-6:
                    slip = ti.min(self.friction_f[None], vt_len / (ti.abs(vn_rel) + 1e-4))
                    v_rel = vn_rel_new * n + (1.0 - slip) * vt_rel
                else:
                    v_rel = vn_rel_new * n
                v = v_rel + vknife
                
        elif which == 2:
            # Knife ↔ Board (use board's restitution/friction)
            if ti.static(self._board != None):
                n = self._board.normal(pos)
                vn = v.dot(n)
                if vn < 0.0:
                    # Apply restitution (energy loss) and friction
                    vn_new = - self._board.restitution_f[None] * vn
                    vt = v - vn * n
                    vt *= (1.0 - self._board.friction_f[None])
                    v = vt + vn_new * n
            # else: no board → nothing
        return v


@ti.data_oriented
class BoardCollider(BaseCollider):
    """Static chopping board. Only collides, does not cut."""
    def __init__(self, sdf_grid, origin, voxel_size, friction=0.5, restitution=0.0):
        super().__init__(sdf_grid, origin, voxel_size, friction, restitution)
        # --- environment hooks & counters ---
        self.has_board_flag = ti.field(dtype=ti.i32, shape=()); self.has_board_flag[None] = 0
        self._hit_mesh = ti.field(dtype=ti.i32, shape=()); self._hit_mesh[None] = 0
        self._hit_board = ti.field(dtype=ti.i32, shape=()); self._hit_board[None] = 0

    @ti.func
    def velocity_at(self, p):
        return ti.Vector([0.0, 0.0, 0.0])

    def update(self, dt: float):
        pass


class MultiCollider:
    """Thin container that keeps named colliders and returns the most penetrating one at a point."""
    def __init__(self, **named):
        self.named = named

    def get(self, name: str):
        return self.named.get(name)

    @property
    def all(self):
        return list(self.named.values())



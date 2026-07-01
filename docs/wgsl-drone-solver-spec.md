# WGSL Batched Drone Solver — Build Spec v1

A batched, fixed-step, deterministic quadrotor dynamics engine in WGSL/WebGPU. One compute dispatch steps `N` independent vehicle environments. Designed so the controller contract (`state in → command out`) is identical in sim and on hardware, and so the full state trajectory is reproducible by SHA-256 regression.

---

## 0. Scope & non-goals

**In:** rigid-body 6-DOF flight, lumped rotor thrust/torque model with motor lag, quadratic body drag, gravity, crash-type collision, batched across `N` envs, per-env domain randomization, deterministic stepping.

**Out (v1):** contact-rich dynamics / friction-cone solving (this is the whole reason we skip MuJoCo), soft bodies, blade-element aerodynamics, inter-rotor downwash coupling, fluid wake. Downwash matters for tight swarms — flag it as a later fidelity tier, not v1.

The discipline: a quadrotor is a free-flying rigid body plus a 4-input force/torque source. That is a small, closed kernel set. Keep it that way.

---

## 1. State vector

Per vehicle, 17 floats (or fixed-point words — see §6):

| Field | Symbol | Dim | Frame | Notes |
|---|---|---|---|---|
| Position | p | 3 | world | z-up |
| Velocity | v | 3 | world | |
| Attitude | q | 4 | — | unit quaternion, Hamilton (w,x,y,z); avoids gimbal lock, cheap to renormalize |
| Angular velocity | ω | 3 | body | |
| Rotor speeds | Ω₁..Ω₄ | 4 | — | first-order motor lag state |

Quaternion over Euler angles is deliberate: no singularities, and renormalization is a single cheap op per step that keeps the integrator stable — both matter for long deterministic rollouts.

Per-env **parameters** live in a separate buffer (so they can be randomized per env for sim-to-real robustness): mass `m`, inertia diag `(Ixx,Iyy,Izz)`, arm length `L`, thrust coeff `k_f`, moment coeff `k_m`, motor time constant `τ_m`, drag coeff `C_d`, rotor geometry `(r_x,r_y,spin)₁..₄`. The obstacle scene is **shared** across envs (one static set), which keeps collision cheap and is fine for coursework.

---

## 2. Dynamics

Rotor `i`: thrust `T_i = k_f · Ω_i²`, reaction torque `Q_i = k_m · Ω_i²`, spin sign `s_i ∈ {+1,−1}`.

**Motor lag** (first-order): `dΩ_i/dt = (Ω_cmd_i − Ω_i) / τ_m`, where `Ω_cmd_i` is the controller's command mapped through the motor curve.

**Body force/torque allocation** (general sum form; X-config is the instance with motors at `(±L/√2, ±L/√2)`):
```
F_z,body = Σ T_i
τ_x      = Σ ( r_iy · T_i )      // roll
τ_y      = Σ ( −r_ix · T_i )     // pitch
τ_z      = Σ ( s_i · Q_i )       // yaw, from reaction torque
```

**Translational** (world frame):
```
a   = (1/m) [ R(q)·(0,0,F_z,body) + (0,0,−m g) − C_d · v·|v| ]
dp  = v
dv  = a
```
`R(q)` rotates body→world. Drag is lumped quadratic; rotate into body frame if you want anisotropy later.

**Rotational** (body frame, Euler's equation):
```
dω = I⁻¹ [ τ − ω × (I·ω) ]
```
`I = diag(Ixx,Iyy,Izz)`, so `I⁻¹` is trivial and `ω×(I·ω)` is one cross product.

**Attitude kinematics** (quaternion):
```
dq = ½ · q ⊗ (0, ω)
q  ← normalize(q)        // every step
```

---

## 3. Integrator

**Semi-implicit (symplectic) Euler, fixed timestep.** One force eval per step, better energy behavior than explicit Euler, fully deterministic. Update velocity before position, ω before q:

```
1. Ω_i += dt·(Ω_cmd_i − Ω_i)/τ_m
2. compute T_i, Q_i, F_z, τ
3. v  += dt·a ;   p += dt·v          // v first
4. ω  += dt·I⁻¹(τ − ω×(I·ω))
5. q  += dt·½ q⊗(0,ω) ;  q = normalize(q)
6. collision check → done flag
```

**Timestep discipline:** physics `dt` fixed (≈1 ms / 1 kHz); control runs at an **integer** substep ratio (e.g., 4 physics substeps per 250 Hz control tick). Never variable dt — that alone breaks determinism. RK4 is an optional higher-fidelity tier; it costs 4 evals and complicates determinism slightly (multi-stage accumulation), so it is not the v1 default.

---

## 4. Collision

Drones are free-flight; collision is **crash detection**, not contact resolution. Model the vehicle as a sphere (radius `r`) or capsule.

- **Ground:** `p.z < r` → contact.
- **Static obstacles:** sphere/AABB primitives, shared scene, broadphase via a uniform grid / spatial hash. Iterate in fixed index order (determinism).
- **Inter-drone (swarm):** pairwise distance within an env — naive O(k²) is fine for k ≤ ~50; spatial hash above that. Per-env, so it parallelizes cleanly.

**Default response: terminate-on-contact** — set the env's `done` flag, freeze/zero state, record a collision code. This is the cleanest signal for RL episodes and swarm safety, and it is *why drones dodge the contact-dynamics problem entirely*. Soft impulse/bounce response is an optional later mode.

---

## 5. WGSL / WebGPU architecture

**Layout: Structure-of-Arrays, env index = thread.** One leading "world dimension" exactly like MuJoCo Warp's `MjData`. One compute invocation owns one env.

⚠️ **vec3 alignment gotcha:** in WGSL storage buffers `vec3<f32>` is 16-byte aligned and strides like a `vec4`, which silently wastes memory and misaligns packed structs. Use **flat `array<f32>`** (or `vec4` with explicit padding), not `array<vec3>`. This bites everyone once.

```wgsl
// SoA state buffers (float path; fixed-point path swaps f32→i32, see §6)
@group(0) @binding(0) var<storage, read_write> s_pos : array<f32>; // 3*N
@group(0) @binding(1) var<storage, read_write> s_vel : array<f32>; // 3*N
@group(0) @binding(2) var<storage, read_write> s_quat: array<f32>; // 4*N
@group(0) @binding(3) var<storage, read_write> s_omg : array<f32>; // 3*N
@group(0) @binding(4) var<storage, read_write> s_rot : array<f32>; // 4*N
@group(0) @binding(5) var<storage, read>       cmd   : array<f32>; // 4*N motor commands
@group(0) @binding(6) var<storage, read>       prm   : array<f32>; // per-env params
@group(0) @binding(7) var<storage, read_write> done  : array<u32>; // N
@group(0) @binding(8) var<uniform>             sim   : SimConst;    // dt, substeps, scene ptrs

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sim.n_env) { return; }
  if (done[i] != 0u) { return; }      // skip finished envs, fixed control flow

  // load env i state + params (offsets = stride*i)
  // ... semi-implicit Euler from §3 ...
  // write back in place (safe: no cross-env reads)
}
```

**Why this is the determinism enabler:** because each env is integrated by a *single thread with no cross-thread reduction*, you never touch the non-deterministic float-atomic-accumulation problem. There is no `atomicAdd` over a shared accumulator anywhere in the step. Per-env trajectories are computed by sequential ops in one invocation.

**Host loop (TS):** per control tick → write `cmd` (from the WASM controllers) → dispatch `K` substeps of `step` → optionally read back state. **Minimize readback** (the real bottleneck): keep state on GPU, read back only what controllers/renderer need. If controllers are CPU-side WASM, the per-tick readback caps practical env count — measure it on the cohort-baseline GPU. Pure-GPU RL keeps everything resident.

**Pipeline:** cache the compute pipeline; **warm-compile** the shader during idle — WebGPU shader compilation is async and cold first-frame compile can exceed 200 ms.

**Rendering:** decoupled. The renderer (Three.js r171 WebGPU, or wgpu) does an instanced draw reading `s_pos`/`s_quat` directly — one instance per drone, no readback needed if it shares the buffers.

---

## 6. Determinism strategy

Per-env independence (§5) kills *within-device* reduction nondeterminism. The remaining problem is **cross-device** bit-exactness, and it is real: identical WGSL on NVIDIA vs Apple vs AMD can diverge because of

1. **FMA contraction** — `a*b+c` may or may not fuse, with different rounding, per hardware/compiler.
2. **Transcendentals** — WGSL does **not** guarantee bit-exact `sin`/`cos`/`sqrt`/`exp` across vendors.
3. **Rounding mode / denormal handling** differences.

In a chaotic system these tiny deltas compound, so "same code" ≠ "same trajectory" across GPUs.

**Target (principled, matches the verifiability thesis): fixed-point.** Integer math is exact and identical on all hardware.
- State and dynamics in **Q-format** (e.g., Q16.16 in `i32`).
- Transcendentals as **your own** deterministic approximations: CORDIC for trig, Newton–Raphson for `sqrt`/reciprocal — no hardware intrinsics in the determinism-critical path.
- ⚠️ **WGSL constraint to verify:** core WGSL has **no native 64-bit integers**. A Q16.16 × Q16.16 multiply needs a 64-bit intermediate before the shift-back, so you must **emulate 64-bit multiply** from 32-bit hi/lo parts (or accept reduced range/precision). Check whether a 64-bit-int extension is enabled on the cohort baseline — if it is, this path gets much cheaper. This single constraint is what makes "bit-exact across devices" a weeks-vs-months call; surface it before committing.

**Pragmatic fallback (faster, not bit-exact across vendors): f32** with FMA contraction disabled where the compiler allows and all transcendentals replaced by controlled polynomials. Deterministic enough *within* a vendor/driver; use a CPU/fixed-point reference as ground truth for grading. Fine for tiers that don't need cross-device exactness.

**Also required regardless of path:**
- Fixed `dt`, fixed substep count, **fixed loop bounds** (no data-dependent iteration).
- Any randomness (domain randomization, sensor noise) from a **counter-based PRNG** (Philox/PCG) seeded by `(env_id, step, stream)` — reproducible per env per step, no sequential state.
- Deterministic ordering of per-env operations (obstacle iteration by fixed index).

---

## 7. Control seam (sim ↔ hardware)

The contract that must hold **identically** in both worlds:

```
observe(state_i)  →  command_i      // 4 normalized motor cmds, or a setpoint an onboard loop converts
```

- **Sim:** `state_i` is read from the SoA buffers (ground truth, optionally + noise/delay model); `command_i` written to `cmd`.
- **Hardware:** `state_i` is the EKF/telemetry estimate; `command_i` goes to the ESCs via the flight controller / MAVLink Offboard.

The **same compiled WASM controller** consumes the same observation layout and emits the same command layout in both. Only the I/O shim differs. That is what makes "develop in browser, deploy to metal" the same artifact rather than a port.

Observation tiers: v1 hands controllers ground-truth state (+ optional additive noise) so the focus stays on control/coordination; a sensor-noise + delay + EKF tier comes later to force estimation.

---

## 8. Validation

- **Reference model:** a CPU implementation (Rust or TS) of §2–3, checked against analytics — hover equilibrium `Σ T_i = m g`, known step responses. This is ground truth for the GPU port.
- **Conservation:** energy/momentum in motors-off ballistic phases.
- **Determinism regression:** run a fixed scenario, **SHA-256 the full state trajectory**, assert identical across runs and across target devices. The fixed-point path must hash identically on NVIDIA/Apple/AMD; the f32 path will not — document the divergence as a known property, not a bug.

---

## 9. Build sequence

1. **CPU reference** — single quad, semi-implicit Euler, validated against hover/step analytics. Ground truth.
2. **Single-env WGSL (f32)** — same math on GPU, one env, match the CPU reference. Validates dynamics independent of determinism.
3. **Batched WGSL (f32)** — `N` envs, SoA, one dispatch steps all. Measure env-count scaling at real-time on the cohort-baseline GPU; measure the readback cost with WASM controllers in the loop.
4. **Determinism hardening** — swap the core to fixed-point (Q-format + emulated 64-bit mul + CORDIC/Newton transcendentals + counter PRNG); stand up the SHA-256 cross-device regression.
5. **Control seam** — wire the WASM controller contract (state in / command out) so the same module runs sim now and hardware later.

Ship 1–3 fast (they're standard and de-risk the physics); 4 is the research-grade differentiator and the part worth doing carefully; 5 is what unifies the two domains.

---

**One open decision before coding:** verify whether a 64-bit-int WGSL extension is available on your cohort's device baseline. If yes, fixed-point is cheap and §6's hard part mostly evaporates. If no, decide up front whether bit-exact cross-device determinism is a v1 requirement (→ emulate 64-bit mul) or a v2 goal (→ ship f32 + fixed-point reference first). That choice sets the difficulty of the whole build.

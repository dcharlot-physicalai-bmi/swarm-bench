<div align="center">

# Swarm Bench

**A deterministic, batched, in-browser drone-swarm physics engine — with a one-binary sim-to-real seam.**

*Batched quadrotor dynamics in WGSL/WebGPU: thousands of independent vehicles stepped in one GPU dispatch, bit-exact across GPUs via fixed-point, with a single compiled control module that flies the browser sim and a real aircraft unchanged.*

[![License: CC0-1.0](https://img.shields.io/badge/license-CC0_1.0-46e0c0?style=flat-square)](LICENSE)
[![Rust + WGSL](https://img.shields.io/badge/Rust_+_WGSL-WebGPU-cfaa5b?style=flat-square)](https://www.rust-lang.org)
[![Physical AI @ BMI](https://img.shields.io/badge/Physical_AI-%40_BMI-cfaa5b?style=flat-square)](https://physicalai-bmi.org)
[![Research topic](https://img.shields.io/badge/topic-AI_%2B_UAS_Swarm_Bench-46e0c0?style=flat-square)](https://physicalai-bmi.org/research/hiner-lab#topic-swarm)

</div>

---

## Why this exists

Teaching and researching drone autonomy shouldn't require a hangar — or a datacenter. **Nobody had shipped batched drone dynamics on WebGPU.** This is that: a quadrotor swarm that runs in a web browser, on the GPU already in the laptop, at zero install — and the *same* controller you develop against the sim compiles to the real aircraft.

It is released **CC0 (public domain)** as a timestamped prior-art commons: the specs and the working reference are here so the ideas stay open.

Two claims carry the whole thing:

1. **Determinism.** The swarm's full trajectory is **fixed-point (Q32.32)** and **bit-exact across GPUs** — hashed with SHA-256, so a run can be replayed, graded, and trusted. (Two independent integer implementations — native `i64`/`i128` on the CPU and *emulated 64-bit* in WGSL on the GPU — produce the same 1500-step rollout, digest `4c5ab55b…`.)
2. **One binary, sim → metal.** A single dependency-free `control()` (state → command) runs identically in the browser sim and on a Pixhawk over MAVLink. Only the I/O shim differs. "Develop in the browser, deploy to metal" as one artifact, not a port.

## The build, by milestone

| Crate | Milestone | What it is | Status |
|-------|-----------|------------|--------|
| [`drone-ref`](crates/drone-ref) | **M1** | Dependency-free **f64 CPU reference** — the behavioral ground truth. | ✅ 5/5 analytic tests (hover, step, convergence, rotation, determinism) |
| [`drone-verify`](crates/drone-verify) | **M2** | Deps-free logic check: same kernel at f64 vs f32 (16 µm over a climb-and-roll). | ✅ builds |
| [`drone-gpu`](crates/drone-gpu) | **M2 / M3** | **Batched WGSL/WebGPU** solver (`drone_step.wgsl`): `N` envs, one dispatch, SoA field-major single buffer. 4096 envs, perfect per-env isolation, ~1.86 M envs/dispatch ceiling. | ✅ compiles (wgpu) |
| `drone-gpu` bins `fixedpt`, `fixedgpu` | **M4** | The determinism keystone: **Q32.32 fixed-point** with emulated 64-bit multiply (`mul_q32.wgsl`), full step (`fixed_step.wgsl`), GPU-vs-CPU **bit-exact**, SHA-256 anchored. | ✅ compiles |
| [`fixedstep`](crates/fixedstep) | **M4** | Deps-free CPU integer reference for the fixed-point step (native `i128`). | ✅ builds |
| [`flightctl`](crates/flightctl) | **M5** | The **control seam**: one `control()` → `flightctl.wasm` (no libm, self-contained `fsqrt`); `bin/sim` flies it closed-loop; [`hardware_shim.rs`](crates/flightctl/hardware_shim.rs) is the MAVLink side. | ✅ builds; closed-loop flight |

**Design facts settled by the build** (see [`docs/wgsl-drone-solver-spec.md`](docs/wgsl-drone-solver-spec.md)): WebGPU caps storage buffers at 8/stage → field-major single buffer; **Q16.16 overflows Ω²** (≈1.2 M) so Q32.32 is mandatory; rounding pinned to truncate-toward-zero; small coefficients fused (`kf·Ω²`); the lone runtime transcendental is a division-free `frsqrt`.

## Layout

```
crates/
  drone-ref/      M1 — f64 CPU reference (+ tests)
  drone-verify/   M2 — f64-vs-f32 logic check
  drone-gpu/      M2/M3/M4 — WGSL/WebGPU batched solver + fixed-point
    src/*.wgsl    drone_step / mul_q32 / fixed_step
    src/bin/      fixedpt, fixedgpu ; src/fixed_ref.rs (shared)
  fixedstep/      M4 — CPU integer step reference
  flightctl/      M5 — one control() for sim + hardware
docs/
  wgsl-drone-solver-spec.md        the canonical build spec
  ai-uas-course-architecture.md    the NDAA-clean AI+UAS course this engine is the spine of
```

## Build

```bash
cargo test  -p drone-ref     # M1 ground truth (5/5)
cargo build -p drone-gpu     # WGSL/WebGPU batched solver (needs wgpu)
cargo run   -p drone-gpu --bin fixedgpu   # M4: prints the trajectory SHA-256
```

> `drone-gpu` pins `wgpu = "0.19"` (verified on software Vulkan). On a current toolchain bump `wgpu` to the latest; the shader and the kernel logic are unchanged.

## Cross-device determinism — results

The fixed-point path is meant to be bit-exact on *every* GPU. Confirmed so far, all producing the same digest:

| Compute | Backend | Trajectory SHA-256 |
|---|---|---|
| CPU native `i64`/`i128` | — | `4c5ab55b…` ✅ |
| GPU emulated-64-bit | software Vulkan (llvmpipe) | `4c5ab55b…` ✅ |
| GPU emulated-64-bit | **Apple M5 Max (Metal)** | `4c5ab55b…` ✅ |
| GPU emulated-64-bit | **NVIDIA A10 (Vulkan, Linux)** | `4c5ab55b…` ✅ |
| GPU emulated-64-bit | AMD (Vulkan/DX12) | *pending — run `fixedgpu`* |

Four independent computations — a native-integer CPU and an *emulated*-64-bit GPU kernel across **three different backends spanning two real GPU vendors (Apple + NVIDIA) and two graphics APIs (Metal + Vulkan)** — produce a byte-for-byte identical 1500-step trajectory (full digest `4c5ab55b05dbc48b94146d5c54e10d37c5253cbf37bec2cf64b068a36f0f5add`). Running `fixedgpu` on AMD is the remaining check; a matching digest closes cross-vendor exactness, and any mismatch localizes to a byte offset.

> The GPU bins select `wgpu::Backends::PRIMARY`, so `fixedgpu` runs on Metal (macOS), Vulkan (Linux), or DX12 (Windows) — whatever the machine has.

## Context

Swarm Bench is the engine behind the **AI + UAS** research topic in the **Hiner Lab** at the Institute for Physical AI, built on the **Charlot Lab's** WebGPU/WASM stack — the *Perceive → Simulate → Act* loop with a deterministic core.

- **Research topic** — https://physicalai-bmi.org/research/hiner-lab#topic-swarm
- **Institute for Physical AI** — https://physicalai-bmi.org

---

<div align="center">
<sub>Released CC0 1.0 (public domain) · Institute for Physical AI · Bailey Military Institute</sub>
</div>

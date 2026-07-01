// Batched quadrotor step kernel — port of the Rust reference (wgsl-drone-solver-spec §3).
// One invocation = one environment (env id = global_invocation_id.x).
//
// Storage layout is chosen for WebGPU portability + GPU coalescing:
//   * exactly THREE storage buffers (WebGPU guarantees only 8 per stage).
//   * `state` is FIELD-MAJOR: value of field f for env i lives at state[f*N + i],
//     so adjacent threads touch adjacent addresses (coalesced) while still fitting
//     one binding. 18 fields per env (see indices below); `done` is field 17 as f32.
// Quaternion convention: (x,y,z,w). Host converts to/from the f64 reference (w,x,y,z).

struct Config {
    m: f32, g: f32, ix: f32, iy: f32, iz: f32,
    kf: f32, km: f32, tau_m: f32, cd: f32, radius: f32,
    rx0: f32, ry0: f32, rx1: f32, ry1: f32, rx2: f32, ry2: f32, rx3: f32, ry3: f32,
    s0: f32, s1: f32, s2: f32, s3: f32,
    dt: f32,
    n_env: u32,
    n_substeps: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>; // field-major, 18*N
@group(0) @binding(1) var<storage, read>       cmd:   array<f32>; // field-major, 4*N
@group(0) @binding(2) var<storage, read>       cfg:   Config;

// field indices (× n_env + env_id)
const PX = 0u; const PY = 1u; const PZ = 2u;
const VX = 3u; const VY = 4u; const VZ = 5u;
const QX = 6u; const QY = 7u; const QZ = 8u; const QW = 9u;
const WX = 10u; const WY = 11u; const WZ = 12u;
const R0 = 13u; const R1 = 14u; const R2 = 15u; const R3 = 16u;
const DONE = 17u;

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let av = a.xyz; let aw = a.w;
    let bv = b.xyz; let bw = b.w;
    return vec4<f32>(aw * bv + bw * av + cross(av, bv), aw * bw - dot(av, bv));
}
fn qrotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz; let s = q.w;
    let t = 2.0 * cross(u, v);
    return v + s * t + cross(u, t);
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = cfg.n_env;
    if (i >= n) { return; }
    if (state[DONE * n + i] != 0.0) { return; }

    var p = vec3<f32>(state[PX*n+i], state[PY*n+i], state[PZ*n+i]);
    var v = vec3<f32>(state[VX*n+i], state[VY*n+i], state[VZ*n+i]);
    var q = vec4<f32>(state[QX*n+i], state[QY*n+i], state[QZ*n+i], state[QW*n+i]);
    var w = vec3<f32>(state[WX*n+i], state[WY*n+i], state[WZ*n+i]);
    var r = vec4<f32>(state[R0*n+i], state[R1*n+i], state[R2*n+i], state[R3*n+i]);
    let c = vec4<f32>(cmd[0u*n+i], cmd[1u*n+i], cmd[2u*n+i], cmd[3u*n+i]);

    let dt = cfg.dt;
    var crashed = false;

    for (var k: u32 = 0u; k < cfg.n_substeps; k = k + 1u) {
        r = r + dt * (c - r) / cfg.tau_m;

        let th = vec4<f32>(cfg.kf*r.x*r.x, cfg.kf*r.y*r.y, cfg.kf*r.z*r.z, cfg.kf*r.w*r.w);
        let qd = vec4<f32>(cfg.km*r.x*r.x, cfg.km*r.y*r.y, cfg.km*r.z*r.z, cfg.km*r.w*r.w);
        let fz = th.x + th.y + th.z + th.w;
        let tx = cfg.ry0*th.x + cfg.ry1*th.y + cfg.ry2*th.z + cfg.ry3*th.w;
        let ty = -(cfg.rx0*th.x + cfg.rx1*th.y + cfg.rx2*th.z + cfg.rx3*th.w);
        let tz = cfg.s0*qd.x + cfg.s1*qd.y + cfg.s2*qd.z + cfg.s3*qd.w;
        let tau = vec3<f32>(tx, ty, tz);

        let thrust_world = qrotate(q, vec3<f32>(0.0, 0.0, fz));
        let speed = length(v);
        let drag = v * (-cfg.cd * speed);
        let accel = (thrust_world + drag) / cfg.m + vec3<f32>(0.0, 0.0, -cfg.g);
        v = v + dt * accel;
        p = p + dt * v;

        let iw = vec3<f32>(cfg.ix*w.x, cfg.iy*w.y, cfg.iz*w.z);
        let net = tau - cross(w, iw);
        let wdot = vec3<f32>(net.x/cfg.ix, net.y/cfg.iy, net.z/cfg.iz);
        w = w + dt * wdot;

        let wq = vec4<f32>(w.x, w.y, w.z, 0.0);
        let dq = qmul(q, wq);
        q = normalize(q + 0.5 * dt * dq);

        if (p.z < cfg.radius) { crashed = true; break; }
    }

    state[PX*n+i]=p.x; state[PY*n+i]=p.y; state[PZ*n+i]=p.z;
    state[VX*n+i]=v.x; state[VY*n+i]=v.y; state[VZ*n+i]=v.z;
    state[QX*n+i]=q.x; state[QY*n+i]=q.y; state[QZ*n+i]=q.z; state[QW*n+i]=q.w;
    state[WX*n+i]=w.x; state[WY*n+i]=w.y; state[WZ*n+i]=w.z;
    state[R0*n+i]=r.x; state[R1*n+i]=r.y; state[R2*n+i]=r.z; state[R3*n+i]=r.w;
    if (crashed) { state[DONE*n+i] = 1.0; }
}

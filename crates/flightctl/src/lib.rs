//! Quadrotor control module — the seam.
//!
//! ONE compiled artifact runs in both places. The ABI is a flat C function over
//! linear memory: `control(state, setpoint, out)`. In the browser sim the host
//! fills `state` from the physics solver and feeds `out` back into `step`; on the
//! aircraft the host fills `state` from MAVLink telemetry and ships `out` as a
//! setpoint. Only that shim differs — this function is byte-identical in both.
//!
//! Self-contained: no_std on wasm, and its own `fsqrt`, so the artifact needs no
//! libm / no host math import — a genuinely portable work unit.
//!
//! Build for the aircraft/browser:
//!   rustup target add wasm32-unknown-unknown
//!   cargo build --release --lib --target wasm32-unknown-unknown
//!   -> target/wasm32-unknown-unknown/release/flightctl.wasm  (exports `control`)

#![cfg_attr(target_arch = "wasm32", no_std)]

// ---- airframe + gains (must match the plant this controller flies) ----
const M: f32 = 0.5;
const G: f32 = 9.81;
const KF: f32 = 1.0e-6;
const KMKF: f32 = 0.016; // km/kf
const ARM: f32 = 0.120208; // 0.17 / sqrt(2)
const IX: f32 = 3.2e-3;
const IY: f32 = 3.2e-3;
const IZ: f32 = 5.5e-3;

const KP_POS: f32 = 6.0;   // position -> desired accel
const KD_POS: f32 = 4.5;   // velocity damping
const KP_ATT: f32 = 10.0;  // tilt error -> desired body rate
const KRATE: f32 = 28.0;   // rate error -> angular accel (x inertia -> torque)
const OMEGA_MAX: f32 = 1600.0;

// self-contained sqrt (bit-hack seed + 3 Newton steps); core-only, no libm.
fn fsqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let i = 0x1fbd_1df5u32.wrapping_add(x.to_bits() >> 1);
    let mut y = f32::from_bits(i);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

// rotate body-frame vector into world by quaternion (x,y,z,w)
fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let cx = u[1] * v[2] - u[2] * v[1];
    let cy = u[2] * v[0] - u[0] * v[2];
    let cz = u[0] * v[1] - u[1] * v[0];
    let (tx, ty, tz) = (2.0 * cx, 2.0 * cy, 2.0 * cz);
    let c2x = u[1] * tz - u[2] * ty;
    let c2y = u[2] * tx - u[0] * tz;
    let c2z = u[0] * ty - u[1] * tx;
    [v[0] + s * tx + c2x, v[1] + s * ty + c2y, v[2] + s * tz + c2z]
}
fn qconj(q: [f32; 4]) -> [f32; 4] { [-q[0], -q[1], -q[2], q[3]] }

/// The seam. Fixed ABI, pure function of its inputs.
/// state[17]    = [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz, r0,r1,r2,r3]
/// setpoint[4]  = [x_des, y_des, z_des, yaw_des]   (yaw held by rate damping here)
/// out[4]       = commanded rotor speeds [Ω0..Ω3]
#[unsafe(no_mangle)]
pub extern "C" fn control(state: *const f32, setpoint: *const f32, out: *mut f32) {
    let s = unsafe { core::slice::from_raw_parts(state, 17) };
    let sp = unsafe { core::slice::from_raw_parts(setpoint, 4) };
    let o = unsafe { core::slice::from_raw_parts_mut(out, 4) };

    let p = [s[0], s[1], s[2]];
    let v = [s[3], s[4], s[5]];
    let q = [s[6], s[7], s[8], s[9]];
    let w = [s[10], s[11], s[12]];

    // 1. position loop -> desired world acceleration (gravity feedforward)
    let ep = [sp[0] - p[0], sp[1] - p[1], sp[2] - p[2]];
    let ades = [
        KP_POS * ep[0] - KD_POS * v[0],
        KP_POS * ep[1] - KD_POS * v[1],
        KP_POS * ep[2] - KD_POS * v[2] + G,
    ];

    // 2. desired body-z (thrust direction) + total thrust along current body-z
    let an = fsqrt(ades[0] * ades[0] + ades[1] * ades[1] + ades[2] * ades[2]);
    let zdes = if an > 1e-6 { [ades[0] / an, ades[1] / an, ades[2] / an] } else { [0.0, 0.0, 1.0] };
    let zcur = qrot(q, [0.0, 0.0, 1.0]);
    let tt = M * (ades[0] * zcur[0] + ades[1] * zcur[1] + ades[2] * zcur[2]);

    // 3. tilt error (world) -> body -> desired rate
    let tew = [
        zcur[1] * zdes[2] - zcur[2] * zdes[1],
        zcur[2] * zdes[0] - zcur[0] * zdes[2],
        zcur[0] * zdes[1] - zcur[1] * zdes[0],
    ];
    let eb = qrot(qconj(q), tew);
    let wdes = [KP_ATT * eb[0], KP_ATT * eb[1], KP_ATT * eb[2]]; // yaw error ~0 -> rate damps yaw

    // 4. rate loop -> torque
    let txq = IX * KRATE * (wdes[0] - w[0]);
    let tyq = IY * KRATE * (wdes[1] - w[1]);
    let tzq = IZ * KRATE * (wdes[2] - w[2]);

    // 5. control allocation: per-rotor thrust f_i from (thrust, torques), then Ω_i = sqrt(f_i/kf)
    let bb = txq / ARM;
    let cc = -tyq / ARM;
    let dd = tzq / KMKF;
    let ff = [
        0.25 * (tt + bb + cc + dd),
        0.25 * (tt - bb - cc + dd),
        0.25 * (tt - bb + cc - dd),
        0.25 * (tt + bb - cc - dd),
    ];
    for i in 0..4 {
        let fi = if ff[i] > 0.0 { ff[i] } else { 0.0 };
        let om = fsqrt(fi / KF);
        o[i] = if om > OMEGA_MAX { OMEGA_MAX } else { om };
    }
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

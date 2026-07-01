//! WGSL Drone Solver — Milestone 1: CPU reference model (f64)
//!
//! Canonical, dependency-free reference implementation of the quadrotor
//! dynamics in wgsl-drone-solver-spec.md (§2–3). This f64 model is the
//! BEHAVIORAL ground truth: the WGSL port — and later the fixed-point
//! determinism path — is validated to match this within a tolerance ε.
//!
//!   cargo run    -> demo rollout + a trajectory fingerprint + hover residual
//!   cargo test   -> hover / vertical-step / convergence / rotational checks
//!
//! Conventions: world frame z-up, body z = thrust axis, Hamilton quaternion
//! q = (w,x,y,z) unit, semi-implicit (symplectic) Euler, fixed dt. The vector
//! and quaternion ops are intentionally tiny and explicit so the kernel ports
//! 1:1 to WGSL (no nalgebra hiding the math).

// ----------------------------- math (portable to WGSL) -----------------------------

#[derive(Clone, Copy, Debug)]
struct V3 { x: f64, y: f64, z: f64 }

impl V3 {
    const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    const fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    fn add(self, o: V3) -> V3 { V3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
    fn sub(self, o: V3) -> V3 { V3::new(self.x - o.x, self.y - o.y, self.z - o.z) }
    fn scale(self, s: f64) -> V3 { V3::new(self.x * s, self.y * s, self.z * s) }
    fn dot(self, o: V3) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    fn cross(self, o: V3) -> V3 {
        V3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn norm(self) -> f64 { self.dot(self).sqrt() }
}

#[derive(Clone, Copy, Debug)]
struct Quat { w: f64, x: f64, y: f64, z: f64 }

impl Quat {
    const fn identity() -> Self { Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 } }
    fn norm(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn normalized(self) -> Quat {
        let n = self.norm();
        let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
        Quat { w: self.w * inv, x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }
    /// Hamilton product a ⊗ b
    fn mul(self, b: Quat) -> Quat {
        Quat {
            w: self.w * b.w - self.x * b.x - self.y * b.y - self.z * b.z,
            x: self.w * b.x + self.x * b.w + self.y * b.z - self.z * b.y,
            y: self.w * b.y - self.x * b.z + self.y * b.w + self.z * b.x,
            z: self.w * b.z + self.x * b.y - self.y * b.x + self.z * b.w,
        }
    }
    /// Rotate a body-frame vector into world: v' = v + 2w(u×v) + 2u×(u×v)
    fn rotate(self, v: V3) -> V3 {
        let u = V3::new(self.x, self.y, self.z);
        let t = u.cross(v).scale(2.0);
        v.add(t.scale(self.w)).add(u.cross(t))
    }
}

// --------------------------------- parameters --------------------------------------

#[derive(Clone, Copy)]
struct Params {
    m: f64,                       // mass (kg)
    g: f64,                       // gravity (m/s^2)
    inertia: V3,                  // diagonal inertia (Ixx,Iyy,Izz) kg·m^2
    kf: f64,                      // thrust coeff:  T = kf * Ω^2
    km: f64,                      // moment coeff:  Q = km * Ω^2
    tau_m: f64,                   // motor time constant (s)
    cd: f64,                      // lumped quadratic drag coeff (0 for analytic tests)
    rotor_pos: [(f64, f64); 4],   // (rx, ry) body-frame rotor positions
    spin: [f64; 4],               // +1 / -1 reaction-torque sign
    radius: f64,                  // collision sphere radius (m)
}

impl Params {
    /// Representative small quad. PLACEHOLDERS — replace with the real airframe.
    fn reference() -> Self {
        let l = 0.17;
        let a = l / std::f64::consts::SQRT_2;
        Params {
            m: 0.5,
            g: 9.81,
            inertia: V3::new(3.2e-3, 3.2e-3, 5.5e-3),
            kf: 1.0e-6,
            km: 1.6e-8,
            tau_m: 0.02,
            cd: 0.1,
            // X-config: corners; +y arm = rotors 0,3 ; -y arm = rotors 1,2
            rotor_pos: [(a, a), (-a, -a), (a, -a), (-a, a)],
            spin: [1.0, 1.0, -1.0, -1.0],
            radius: 0.10,
        }
    }
    /// Per-rotor speed for steady hover.
    fn hover_omega(&self) -> f64 { ((self.m * self.g / 4.0) / self.kf).sqrt() }
    /// Per-rotor speed for a given total thrust.
    fn omega_for_total_thrust(&self, t_total: f64) -> f64 { ((t_total / 4.0) / self.kf).sqrt() }
}

// ----------------------------------- state -----------------------------------------

#[derive(Clone, Copy)]
struct State {
    p: V3,            // position (world)
    v: V3,            // velocity (world)
    q: Quat,          // attitude (body->world)
    w: V3,            // angular velocity (body)
    rotor: [f64; 4],  // rotor speeds (rad/s)
    done: bool,       // collision / termination
}

impl State {
    fn rest_at(p: V3, rotor: f64) -> Self {
        State {
            p,
            v: V3::zero(),
            q: Quat::identity(),
            w: V3::zero(),
            rotor: [rotor; 4],
            done: false,
        }
    }
}

// --------------------------------- dynamics ----------------------------------------

/// Body-frame wrench from current rotor speeds: (total thrust along +z_body, torque).
fn rotor_wrench(s: &State, prm: &Params) -> (f64, V3) {
    let mut fz = 0.0;
    let (mut tx, mut ty, mut tz) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let om = s.rotor[i];
        let thrust = prm.kf * om * om;
        let react = prm.km * om * om;
        let (rx, ry) = prm.rotor_pos[i];
        fz += thrust;
        tx += ry * thrust;        // roll
        ty += -rx * thrust;       // pitch
        tz += prm.spin[i] * react; // yaw (reaction)
    }
    (fz, V3::new(tx, ty, tz))
}

/// One semi-implicit Euler substep (§3 of the spec).
fn step(s: &mut State, prm: &Params, cmd: &[f64; 4], dt: f64) {
    if s.done {
        return;
    }

    // 1. motor lag (first-order)
    for i in 0..4 {
        s.rotor[i] += dt * (cmd[i] - s.rotor[i]) / prm.tau_m;
    }

    // 2. forces & torques at current config
    let (fz_body, tau) = rotor_wrench(s, prm);
    let thrust_world = s.q.rotate(V3::new(0.0, 0.0, fz_body));
    let speed = s.v.norm();
    let drag_world = s.v.scale(-prm.cd * speed); // -Cd |v| v
    let accel = thrust_world
        .add(drag_world)
        .scale(1.0 / prm.m)
        .add(V3::new(0.0, 0.0, -prm.g));

    // 3. translational: velocity first (semi-implicit), then position with v_new
    s.v = s.v.add(accel.scale(dt));
    s.p = s.p.add(s.v.scale(dt));

    // 4. rotational: ω̇ = I⁻¹ (τ − ω×(Iω))
    let iw = V3::new(prm.inertia.x * s.w.x, prm.inertia.y * s.w.y, prm.inertia.z * s.w.z);
    let net = tau.sub(s.w.cross(iw));
    let wdot = V3::new(net.x / prm.inertia.x, net.y / prm.inertia.y, net.z / prm.inertia.z);
    s.w = s.w.add(wdot.scale(dt));

    // 5. attitude: dq = ½ q ⊗ (0, ω_new), integrate, renormalize
    let wq = Quat { w: 0.0, x: s.w.x, y: s.w.y, z: s.w.z };
    let qdot = s.q.mul(wq);
    s.q = Quat {
        w: s.q.w + 0.5 * dt * qdot.w,
        x: s.q.x + 0.5 * dt * qdot.x,
        y: s.q.y + 0.5 * dt * qdot.y,
        z: s.q.z + 0.5 * dt * qdot.z,
    }
    .normalized();

    // 6. crash-type collision (ground plane)
    if s.p.z < prm.radius {
        s.done = true;
    }
}

// ------------------------- trajectory fingerprint (regression anchor) ----------------
// FNV-1a over the f64 bit patterns. Deterministic run-to-run on one machine; the
// fixed-point GPU path (milestone 4) is what gives bit-exactness ACROSS devices.

fn state_words(s: &State) -> [f64; 17] {
    [
        s.p.x, s.p.y, s.p.z, s.v.x, s.v.y, s.v.z, s.q.w, s.q.x, s.q.y, s.q.z, s.w.x, s.w.y,
        s.w.z, s.rotor[0], s.rotor[1], s.rotor[2], s.rotor[3],
    ]
}

fn hash_update(h: &mut u64, vals: &[f64]) {
    for &x in vals {
        for b in x.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

// ------------------------------------- main ----------------------------------------

fn main() {
    let prm = Params::reference();
    let oh = prm.hover_omega();
    let rpm = oh * 60.0 / (2.0 * std::f64::consts::PI);
    println!("reference quad: m = {} kg, hover Ω = {:.1} rad/s ({:.0} RPM/rotor)", prm.m, oh, rpm);

    // demo rollout: 1 s hover, then 1 s at +15% collective, fingerprinted.
    let dt = 1e-3;
    let mut s = State::rest_at(V3::new(0.0, 0.0, 10.0), oh);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for k in 0..2000 {
        let cmd = if k < 1000 { [oh; 4] } else { [oh * 1.15; 4] };
        step(&mut s, &prm, &cmd, dt);
        hash_update(&mut h, &state_words(&s));
    }
    println!("after 1 s hover + 1 s @ +15% collective:");
    println!("  pos = ({:.3}, {:.3}, {:.3}) m", s.p.x, s.p.y, s.p.z);
    println!("  vel = ({:.3}, {:.3}, {:.3}) m/s", s.v.x, s.v.y, s.v.z);
    println!("  trajectory FNV-1a = {:#018x}", h);

    // hover residual (full validation suite: `cargo test`)
    let mut hs = State::rest_at(V3::new(0.0, 0.0, 10.0), oh);
    for _ in 0..2000 {
        step(&mut hs, &prm, &[oh; 4], dt);
    }
    let drift = hs.p.sub(V3::new(0.0, 0.0, 10.0)).norm();
    println!("hover drift over 2 s = {:.2e} m (expect ~0)", drift);
}

// ------------------------------------ tests ----------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(mut s: State, prm: &Params, cmd: [f64; 4], dt: f64, steps: usize) -> State {
        for _ in 0..steps {
            step(&mut s, prm, &cmd, dt);
        }
        s
    }

    #[test]
    fn hover_equilibrium() {
        // Exact hover thrust, level, at rest: net force is zero, so nothing should move.
        let prm = Params::reference();
        let oh = prm.hover_omega();
        let p0 = V3::new(0.0, 0.0, 10.0);
        let s0 = State::rest_at(p0, oh);
        let s = run(s0, &prm, [oh; 4], 1e-3, 2000); // 2 s
        assert!(s.p.sub(p0).norm() < 1e-6, "hover drift = {}", s.p.sub(p0).norm());
        assert!(s.v.norm() < 1e-6, "hover speed = {}", s.v.norm());
        assert!((s.q.w.abs() - 1.0).abs() < 1e-9, "attitude tilted, w = {}", s.q.w);
        assert!(!s.done);
    }

    #[test]
    fn vertical_step_matches_analytic() {
        // Drag off, level, ΣT = 2mg  ->  a_z = +g  ->  v_z = g t, p_z = ½ g t².
        let mut prm = Params::reference();
        prm.cd = 0.0;
        let om = prm.omega_for_total_thrust(2.0 * prm.m * prm.g);
        let z0 = 100.0;
        let s0 = State::rest_at(V3::new(0.0, 0.0, z0), om); // start at target speed: no lag transient
        let dt = 1e-3;
        let t = 1.0;
        let s = run(s0, &prm, [om; 4], dt, (t / dt) as usize);

        // velocity is exact at sample points for constant acceleration
        assert!((s.v.z - prm.g * t).abs() < 1e-9, "v_z = {} vs {}", s.v.z, prm.g * t);

        // position carries the O(dt) semi-implicit offset = ½ g dt t
        let dz = s.p.z - z0;
        let pz_analytic = 0.5 * prm.g * t * t;
        let rel = ((dz - pz_analytic) / pz_analytic).abs();
        assert!(rel < 5e-3, "p_z relative error = {}", rel);

        // no lateral drift
        assert!(s.p.x.abs() < 1e-12 && s.p.y.abs() < 1e-12, "lateral drift");
    }

    #[test]
    fn first_order_convergence() {
        // Position error vs analytic should halve as dt halves (first-order method).
        let mut prm = Params::reference();
        prm.cd = 0.0;
        let om = prm.omega_for_total_thrust(2.0 * prm.m * prm.g);
        let z0 = 1000.0;
        let t = 1.0;
        let err = |dt: f64| {
            let s = run(State::rest_at(V3::new(0.0, 0.0, z0), om), &prm, [om; 4], dt, (t / dt) as usize);
            ((s.p.z - z0) - 0.5 * prm.g * t * t).abs()
        };
        let ratio = err(1e-3) / err(2e-3);
        assert!((ratio - 0.5).abs() < 0.05, "convergence ratio = {} (expect ~0.5)", ratio);
    }

    #[test]
    fn rotational_response() {
        // Pure roll differential -> τx only; check initial ω̇x = τx / Ixx, quaternion stays unit.
        let prm = Params::reference();
        let oh = prm.hover_omega();
        let d = 50.0;
        let cmd = [oh + d, oh - d, oh - d, oh + d]; // raise +y arm, lower -y arm
        let mut s = State::rest_at(V3::new(0.0, 0.0, 100.0), oh);
        s.rotor = cmd; // sit at the command so motor lag contributes nothing this step

        let (_, tau0) = rotor_wrench(&s, &prm);
        assert!(tau0.x > 0.0, "expected positive roll torque, got {}", tau0.x);
        assert!(tau0.y.abs() < 1e-9 && tau0.z.abs() < 1e-9, "differential leaked into pitch/yaw");

        let dt = 1e-4;
        let mut s1 = s;
        step(&mut s1, &prm, &cmd, dt);
        let wdot_x = s1.w.x / dt;
        let expect = tau0.x / prm.inertia.x;
        assert!((wdot_x - expect).abs() < 1e-6 * (1.0 + expect.abs()), "ω̇x = {} vs {}", wdot_x, expect);

        let s2 = run(s, &prm, cmd, dt, 5000);
        assert!((s2.q.norm() - 1.0).abs() < 1e-9, "quaternion norm drifted to {}", s2.q.norm());
    }

    #[test]
    fn fingerprint_is_deterministic() {
        // Same scenario twice -> identical fingerprint (run-to-run determinism on one machine).
        let prm = Params::reference();
        let oh = prm.hover_omega();
        let fp = || {
            let mut s = State::rest_at(V3::new(0.0, 0.0, 10.0), oh);
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for _ in 0..1000 {
                step(&mut s, &prm, &[oh; 4], 1e-3);
                hash_update(&mut h, &state_words(&s));
            }
            h
        };
        assert_eq!(fp(), fp());
    }
}

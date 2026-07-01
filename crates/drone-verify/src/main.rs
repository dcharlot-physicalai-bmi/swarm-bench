//! Milestone 2 — port logic verification (dependency-free).
//!
//! The WGSL kernel in drone-gpu/src/drone_step.wgsl is transcribed here into a
//! single generic body, instantiated at BOTH f64 (reference) and f32 (what the
//! GPU runs). Same quaternion convention (x,y,z,w), same operation order as the
//! shader. Rolling the same scenario through both isolates one thing: does the
//! f32 kernel reproduce the f64 reference? That is exactly the milestone-2
//! question, and it needs no GPU to answer.

#[derive(Clone, Copy)]
pub struct Params {
    pub m: f64,
    pub g: f64,
    pub ix: f64,
    pub iy: f64,
    pub iz: f64,
    pub kf: f64,
    pub km: f64,
    pub tau_m: f64,
    pub cd: f64,
    pub radius: f64,
    pub rpos: [(f64, f64); 4],
    pub spin: [f64; 4],
}

impl Params {
    pub fn reference() -> Self {
        let l = 0.17f64;
        let a = l / std::f64::consts::SQRT_2;
        Params {
            m: 0.5,
            g: 9.81,
            ix: 3.2e-3,
            iy: 3.2e-3,
            iz: 5.5e-3,
            kf: 1.0e-6,
            km: 1.6e-8,
            tau_m: 0.02,
            cd: 0.1,
            radius: 0.10,
            rpos: [(a, a), (-a, -a), (a, -a), (-a, a)],
            spin: [1.0, 1.0, -1.0, -1.0],
        }
    }
    pub fn hover_omega(&self) -> f64 {
        ((self.m * self.g / 4.0) / self.kf).sqrt()
    }
}

// Same kernel body at any float width. `as F` casts the f64 literals down for f32.
macro_rules! make_dyn {
    ($name:ident, $F:ty) => {
        pub mod $name {
            #![allow(dead_code)]
            type F = $F;

            #[derive(Clone, Copy)]
            pub struct V {
                pub x: F,
                pub y: F,
                pub z: F,
            }
            impl V {
                pub fn n(x: F, y: F, z: F) -> V { V { x, y, z } }
                pub fn add(self, o: V) -> V { V::n(self.x + o.x, self.y + o.y, self.z + o.z) }
                pub fn sub(self, o: V) -> V { V::n(self.x - o.x, self.y - o.y, self.z - o.z) }
                pub fn s(self, k: F) -> V { V::n(self.x * k, self.y * k, self.z * k) }
                pub fn dot(self, o: V) -> F { self.x * o.x + self.y * o.y + self.z * o.z }
                pub fn cross(self, o: V) -> V {
                    V::n(
                        self.y * o.z - self.z * o.y,
                        self.z * o.x - self.x * o.z,
                        self.x * o.y - self.y * o.x,
                    )
                }
                pub fn len(self) -> F { self.dot(self).sqrt() }
            }

            // quaternion as [x, y, z, w]
            pub fn qmul(a: [F; 4], b: [F; 4]) -> [F; 4] {
                let av = V::n(a[0], a[1], a[2]);
                let aw = a[3];
                let bv = V::n(b[0], b[1], b[2]);
                let bw = b[3];
                let v = bv.s(aw).add(av.s(bw)).add(av.cross(bv));
                let w = aw * bw - av.dot(bv);
                [v.x, v.y, v.z, w]
            }
            pub fn qrot(q: [F; 4], v: V) -> V {
                let u = V::n(q[0], q[1], q[2]);
                let s = q[3];
                let t = u.cross(v).s(2.0 as F);
                v.add(t.s(s)).add(u.cross(t))
            }
            pub fn qnorm(q: [F; 4]) -> [F; 4] {
                let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                let inv = if n > (0.0 as F) { (1.0 as F) / n } else { 0.0 as F };
                [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
            }

            pub struct P {
                pub m: F,
                pub g: F,
                pub ix: F,
                pub iy: F,
                pub iz: F,
                pub kf: F,
                pub km: F,
                pub tau_m: F,
                pub cd: F,
                pub radius: F,
                pub rpos: [(F, F); 4],
                pub spin: [F; 4],
            }
            pub fn pconv(s: &crate::Params) -> P {
                P {
                    m: s.m as F, g: s.g as F, ix: s.ix as F, iy: s.iy as F, iz: s.iz as F,
                    kf: s.kf as F, km: s.km as F, tau_m: s.tau_m as F, cd: s.cd as F, radius: s.radius as F,
                    rpos: [
                        (s.rpos[0].0 as F, s.rpos[0].1 as F),
                        (s.rpos[1].0 as F, s.rpos[1].1 as F),
                        (s.rpos[2].0 as F, s.rpos[2].1 as F),
                        (s.rpos[3].0 as F, s.rpos[3].1 as F),
                    ],
                    spin: [s.spin[0] as F, s.spin[1] as F, s.spin[2] as F, s.spin[3] as F],
                }
            }

            pub struct St {
                pub p: V,
                pub v: V,
                pub q: [F; 4],
                pub w: V,
                pub r: [F; 4],
                pub done: bool,
            }

            // 1:1 with the WGSL `step` kernel.
            pub fn step(s: &mut St, p: &P, c: [F; 4], dt: F) {
                if s.done {
                    return;
                }
                for i in 0..4 {
                    s.r[i] += dt * (c[i] - s.r[i]) / p.tau_m;
                }
                let th = [
                    p.kf * s.r[0] * s.r[0], p.kf * s.r[1] * s.r[1],
                    p.kf * s.r[2] * s.r[2], p.kf * s.r[3] * s.r[3],
                ];
                let qd = [
                    p.km * s.r[0] * s.r[0], p.km * s.r[1] * s.r[1],
                    p.km * s.r[2] * s.r[2], p.km * s.r[3] * s.r[3],
                ];
                let fz = th[0] + th[1] + th[2] + th[3];
                let tx = p.rpos[0].1 * th[0] + p.rpos[1].1 * th[1] + p.rpos[2].1 * th[2] + p.rpos[3].1 * th[3];
                let ty = -(p.rpos[0].0 * th[0] + p.rpos[1].0 * th[1] + p.rpos[2].0 * th[2] + p.rpos[3].0 * th[3]);
                let tz = p.spin[0] * qd[0] + p.spin[1] * qd[1] + p.spin[2] * qd[2] + p.spin[3] * qd[3];
                let tau = V::n(tx, ty, tz);

                let tw = qrot(s.q, V::n(0.0 as F, 0.0 as F, fz));
                let speed = s.v.len();
                let drag = s.v.s(-p.cd * speed);
                let accel = tw.add(drag).s((1.0 as F) / p.m).add(V::n(0.0 as F, 0.0 as F, -p.g));
                s.v = s.v.add(accel.s(dt));
                s.p = s.p.add(s.v.s(dt));

                let iw = V::n(p.ix * s.w.x, p.iy * s.w.y, p.iz * s.w.z);
                let net = tau.sub(s.w.cross(iw));
                let wdot = V::n(net.x / p.ix, net.y / p.iy, net.z / p.iz);
                s.w = s.w.add(wdot.s(dt));

                let wq = [s.w.x, s.w.y, s.w.z, 0.0 as F];
                let dq = qmul(s.q, wq);
                s.q = qnorm([
                    s.q[0] + (0.5 as F) * dt * dq[0],
                    s.q[1] + (0.5 as F) * dt * dq[1],
                    s.q[2] + (0.5 as F) * dt * dq[2],
                    s.q[3] + (0.5 as F) * dt * dq[3],
                ]);

                if s.p.z < p.radius {
                    s.done = true;
                }
            }
        }
    };
}

make_dyn!(refd, f64);
make_dyn!(mir, f32);

fn main() {
    let params = Params::reference();
    let oh = params.hover_omega();

    // Scenario: +20% collective (climb) with a small roll differential — exercises
    // thrust, drag, rotor torque, gyroscopic coupling, and quaternion integration.
    let d = 30.0;
    let cmd = [oh * 1.2 + d, oh * 1.2 - d, oh * 1.2 - d, oh * 1.2 + d];
    let dt = 1e-3;
    let n = 1500; // 1.5 s

    // f64 reference
    let pr = refd::pconv(&params);
    let mut sr = refd::St {
        p: refd::V::n(0.0, 0.0, 10.0),
        v: refd::V::n(0.0, 0.0, 0.0),
        q: [0.0, 0.0, 0.0, 1.0],
        w: refd::V::n(0.0, 0.0, 0.0),
        r: [oh; 4],
        done: false,
    };
    for _ in 0..n {
        refd::step(&mut sr, &pr, cmd, dt);
    }

    // f32 mirror (identical logic, single precision)
    let pm = mir::pconv(&params);
    let mut sm = mir::St {
        p: mir::V::n(0.0, 0.0, 10.0),
        v: mir::V::n(0.0, 0.0, 0.0),
        q: [0.0, 0.0, 0.0, 1.0],
        w: mir::V::n(0.0, 0.0, 0.0),
        r: [oh as f32; 4],
        done: false,
    };
    let cmf = [cmd[0] as f32, cmd[1] as f32, cmd[2] as f32, cmd[3] as f32];
    for _ in 0..n {
        mir::step(&mut sm, &pm, cmf, dt as f32);
    }

    // compare
    let dp = ((sr.p.x - sm.p.x as f64).powi(2)
        + (sr.p.y - sm.p.y as f64).powi(2)
        + (sr.p.z - sm.p.z as f64).powi(2))
    .sqrt();
    let dv = ((sr.v.x - sm.v.x as f64).powi(2)
        + (sr.v.y - sm.v.y as f64).powi(2)
        + (sr.v.z - sm.v.z as f64).powi(2))
    .sqrt();
    let qr = sr.q;
    let qm = [sm.q[0] as f64, sm.q[1] as f64, sm.q[2] as f64, sm.q[3] as f64];
    let dotq = (qr[0] * qm[0] + qr[1] * qm[1] + qr[2] * qm[2] + qr[3] * qm[3])
        .abs()
        .min(1.0);
    let ang_deg = 2.0 * dotq.acos() * 180.0 / std::f64::consts::PI;

    println!("scenario: 1.5 s, +20% collective + roll differential, dt = {} s", dt);
    println!("f64 reference final:");
    println!("  pos = ({:.4}, {:.4}, {:.4})   vel = ({:.4}, {:.4}, {:.4})", sr.p.x, sr.p.y, sr.p.z, sr.v.x, sr.v.y, sr.v.z);
    println!("f32 kernel  final:");
    println!("  pos = ({:.4}, {:.4}, {:.4})   vel = ({:.4}, {:.4}, {:.4})", sm.p.x, sm.p.y, sm.p.z, sm.v.x, sm.v.y, sm.v.z);
    println!("f32-vs-f64 divergence over 1.5 s:");
    println!("  position   = {:.3e} m", dp);
    println!("  velocity   = {:.3e} m/s", dv);
    println!("  attitude   = {:.3e} deg", ang_deg);

    assert!(dp < 0.05, "position divergence too large: {} m", dp);
    assert!(ang_deg < 0.5, "attitude divergence too large: {} deg", ang_deg);
    println!("PASS: f32 kernel tracks f64 reference within tolerance.");
}

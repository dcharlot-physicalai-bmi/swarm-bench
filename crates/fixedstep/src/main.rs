//! Milestone 4 — full Q32.32 fixed-point quadrotor step (CPU reference algorithm).
//!
//! Every operation is integer-exact: multiply via i128 (truncate toward zero),
//! inverse-sqrt via division-free Newton seeded by bit position. No runtime
//! division (all constant divides are pre-inverted on the host). This is the
//! exact algorithm the WGSL kernel transcribes; because the WGSL emulated
//! multiply already matches i128 bit-for-bit, the two implementations agree
//! bit-for-bit, which is the basis for cross-device reproducibility.
//!
//! Verifies: (1) frsqrt precision, (2) fixed-point trajectory vs f64 reference,
//! (3) SHA-256 of the fixed-point trajectory (the regression anchor), self-checked
//! against known SHA-256 answers.

const FRAC: u32 = 32;
const ONE: i64 = 1 << FRAC;          // 1.0
const HALF: i64 = 1 << (FRAC - 1);   // 0.5
const THREE_HALF: i64 = 3 * HALF;    // 1.5
const TWO: i64 = 2 * ONE;            // 2.0

#[inline]
fn fx(x: f64) -> i64 { (x * (ONE as f64)).round() as i64 }
#[inline]
fn dbl(v: i64) -> f64 { v as f64 / (ONE as f64) }
#[inline]
fn fadd(a: i64, b: i64) -> i64 { a.wrapping_add(b) }
#[inline]
fn fsub(a: i64, b: i64) -> i64 { a.wrapping_sub(b) }
#[inline]
fn fneg(a: i64) -> i64 { a.wrapping_neg() }
#[inline]
fn fmul(a: i64, b: i64) -> i64 { ((a as i128) * (b as i128) / (1i128 << FRAC)) as i64 }

/// 1/sqrt(x) in Q32.32 for x > 0. Division-free Newton; integer seed by MSB.
fn frsqrt(x: i64) -> i64 {
    if x <= 0 { return 0; }
    let e = 63 - (x as u64).leading_zeros() as i32; // index of highest set bit
    let shift = (96 - e) / 2;                        // makes x*y0^2 in [~0.5, 1]
    let mut y = 1i64 << shift;
    for _ in 0..6 {
        let y2 = fmul(y, y);
        let t = THREE_HALF - fmul(HALF, fmul(x, y2)); // 1.5 - 0.5*x*y^2
        y = fmul(y, t);
    }
    y
}

// fixed-point vector / quaternion helpers (quaternion is (x,y,z,w))
fn fcross(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [fsub(fmul(a[1], b[2]), fmul(a[2], b[1])),
     fsub(fmul(a[2], b[0]), fmul(a[0], b[2])),
     fsub(fmul(a[0], b[1]), fmul(a[1], b[0]))]
}
fn fdot3(a: [i64; 3], b: [i64; 3]) -> i64 {
    fadd(fadd(fmul(a[0], b[0]), fmul(a[1], b[1])), fmul(a[2], b[2]))
}
fn fqrot(q: [i64; 4], v: [i64; 3]) -> [i64; 3] {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    let c1 = fcross(u, v);
    let t = [fmul(TWO, c1[0]), fmul(TWO, c1[1]), fmul(TWO, c1[2])];
    let c2 = fcross(u, t);
    [fadd(fadd(v[0], fmul(s, t[0])), c2[0]),
     fadd(fadd(v[1], fmul(s, t[1])), c2[1]),
     fadd(fadd(v[2], fmul(s, t[2])), c2[2])]
}
fn fqmul(a: [i64; 4], b: [i64; 4]) -> [i64; 4] {
    let av = [a[0], a[1], a[2]];
    let aw = a[3];
    let bv = [b[0], b[1], b[2]];
    let bw = b[3];
    let c = fcross(av, bv);
    [fadd(fadd(fmul(aw, bv[0]), fmul(bw, av[0])), c[0]),
     fadd(fadd(fmul(aw, bv[1]), fmul(bw, av[1])), c[1]),
     fadd(fadd(fmul(aw, bv[2]), fmul(bw, av[2])), c[2]),
     fsub(fmul(aw, bw), fdot3(av, bv))]
}

// pre-inverted params (all Q32.32)
struct FxP {
    kf: i64, km: i64, cd: i64, g: i64, radius: i64, dt: i64, half_dt: i64,
    ix: i64, iy: i64, iz: i64,
    inv_tau: i64, inv_m: i64, inv_ix: i64, inv_iy: i64, inv_iz: i64,
    rpos: [(i64, i64); 4], spin: [i64; 4],
}
#[derive(Clone, Copy)]
struct FxS { p: [i64; 3], v: [i64; 3], q: [i64; 4], w: [i64; 3], r: [i64; 4], done: bool }

fn fstep(s: &mut FxS, p: &FxP, cmd: [i64; 4]) {
    if s.done { return; }
    for i in 0..4 {
        let dd = fmul(fmul(p.dt, fsub(cmd[i], s.r[i])), p.inv_tau);
        s.r[i] = fadd(s.r[i], dd);
    }
    let mut th = [0i64; 4];
    let mut re = [0i64; 4];
    for i in 0..4 {
        let r2 = fmul(s.r[i], s.r[i]);
        th[i] = fmul(p.kf, r2);
        re[i] = fmul(p.km, r2);
    }
    let fz = fadd(fadd(th[0], th[1]), fadd(th[2], th[3]));
    let tx = fadd(fadd(fmul(p.rpos[0].1, th[0]), fmul(p.rpos[1].1, th[1])),
                  fadd(fmul(p.rpos[2].1, th[2]), fmul(p.rpos[3].1, th[3])));
    let ty = fneg(fadd(fadd(fmul(p.rpos[0].0, th[0]), fmul(p.rpos[1].0, th[1])),
                       fadd(fmul(p.rpos[2].0, th[2]), fmul(p.rpos[3].0, th[3]))));
    let tz = fadd(fadd(fmul(p.spin[0], re[0]), fmul(p.spin[1], re[1])),
                  fadd(fmul(p.spin[2], re[2]), fmul(p.spin[3], re[3])));
    let tau = [tx, ty, tz];

    let tw = fqrot(s.q, [0, 0, fz]);
    let vv = fadd(fadd(fmul(s.v[0], s.v[0]), fmul(s.v[1], s.v[1])), fmul(s.v[2], s.v[2]));
    let speed = fmul(vv, frsqrt(vv)); // sqrt(vv); 0 when vv==0
    let dfac = fneg(fmul(p.cd, speed));
    let drag = [fmul(dfac, s.v[0]), fmul(dfac, s.v[1]), fmul(dfac, s.v[2])];
    let mut acc = [fmul(fadd(tw[0], drag[0]), p.inv_m),
                   fmul(fadd(tw[1], drag[1]), p.inv_m),
                   fmul(fadd(tw[2], drag[2]), p.inv_m)];
    acc[2] = fsub(acc[2], p.g);
    for k in 0..3 { s.v[k] = fadd(s.v[k], fmul(p.dt, acc[k])); }
    for k in 0..3 { s.p[k] = fadd(s.p[k], fmul(p.dt, s.v[k])); }

    let iw = [fmul(p.ix, s.w[0]), fmul(p.iy, s.w[1]), fmul(p.iz, s.w[2])];
    let gyro = fcross(s.w, iw);
    let net = [fsub(tau[0], gyro[0]), fsub(tau[1], gyro[1]), fsub(tau[2], gyro[2])];
    let wdot = [fmul(net[0], p.inv_ix), fmul(net[1], p.inv_iy), fmul(net[2], p.inv_iz)];
    for k in 0..3 { s.w[k] = fadd(s.w[k], fmul(p.dt, wdot[k])); }

    let wq = [s.w[0], s.w[1], s.w[2], 0];
    let dq = fqmul(s.q, wq);
    let mut q = [fadd(s.q[0], fmul(p.half_dt, dq[0])),
                 fadd(s.q[1], fmul(p.half_dt, dq[1])),
                 fadd(s.q[2], fmul(p.half_dt, dq[2])),
                 fadd(s.q[3], fmul(p.half_dt, dq[3]))];
    let qq = fadd(fadd(fmul(q[0], q[0]), fmul(q[1], q[1])), fadd(fmul(q[2], q[2]), fmul(q[3], q[3])));
    let inv = frsqrt(qq);
    for k in 0..4 { q[k] = fmul(q[k], inv); }
    s.q = q;

    if s.p[2] < p.radius { s.done = true; }
}

// ---------------- f64 reference (identical operation order) ----------------
#[derive(Clone, Copy)]
struct Pf { m: f64, g: f64, ix: f64, iy: f64, iz: f64, kf: f64, km: f64, tau: f64, cd: f64, radius: f64, rpos: [(f64, f64); 4], spin: [f64; 4] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]] }
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]]; let s = q[3];
    let c1 = cross(u, v); let t = [2.0*c1[0], 2.0*c1[1], 2.0*c1[2]]; let c2 = cross(u, t);
    [v[0]+s*t[0]+c2[0], v[1]+s*t[1]+c2[1], v[2]+s*t[2]+c2[2]]
}
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let av = [a[0], a[1], a[2]]; let aw = a[3]; let bv = [b[0], b[1], b[2]]; let bw = b[3];
    let c = cross(av, bv);
    [aw*bv[0]+bw*av[0]+c[0], aw*bv[1]+bw*av[1]+c[1], aw*bv[2]+bw*av[2]+c[2], aw*bw-dot3(av, bv)]
}
#[derive(Clone, Copy)]
struct Sf { p: [f64; 3], v: [f64; 3], q: [f64; 4], w: [f64; 3], r: [f64; 4], done: bool }
fn fstep_f64(s: &mut Sf, p: &Pf, cmd: [f64; 4], dt: f64) {
    if s.done { return; }
    for i in 0..4 { s.r[i] += dt*(cmd[i]-s.r[i])/p.tau; }
    let mut th = [0.0; 4]; let mut re = [0.0; 4];
    for i in 0..4 { let r2 = s.r[i]*s.r[i]; th[i]=p.kf*r2; re[i]=p.km*r2; }
    let fz = th[0]+th[1]+th[2]+th[3];
    let tx = p.rpos[0].1*th[0]+p.rpos[1].1*th[1]+p.rpos[2].1*th[2]+p.rpos[3].1*th[3];
    let ty = -(p.rpos[0].0*th[0]+p.rpos[1].0*th[1]+p.rpos[2].0*th[2]+p.rpos[3].0*th[3]);
    let tz = p.spin[0]*re[0]+p.spin[1]*re[1]+p.spin[2]*re[2]+p.spin[3]*re[3];
    let tau = [tx, ty, tz];
    let tw = qrot(s.q, [0.0, 0.0, fz]);
    let vv = s.v[0]*s.v[0]+s.v[1]*s.v[1]+s.v[2]*s.v[2];
    let speed = vv.sqrt();
    let dfac = -p.cd*speed;
    let drag = [dfac*s.v[0], dfac*s.v[1], dfac*s.v[2]];
    let mut acc = [(tw[0]+drag[0])/p.m, (tw[1]+drag[1])/p.m, (tw[2]+drag[2])/p.m];
    acc[2] -= p.g;
    for k in 0..3 { s.v[k] += dt*acc[k]; }
    for k in 0..3 { s.p[k] += dt*s.v[k]; }
    let iw = [p.ix*s.w[0], p.iy*s.w[1], p.iz*s.w[2]];
    let gyro = cross(s.w, iw);
    let net = [tau[0]-gyro[0], tau[1]-gyro[1], tau[2]-gyro[2]];
    let wdot = [net[0]/p.ix, net[1]/p.iy, net[2]/p.iz];
    for k in 0..3 { s.w[k] += dt*wdot[k]; }
    let wq = [s.w[0], s.w[1], s.w[2], 0.0];
    let dq = qmul(s.q, wq);
    let mut q = [s.q[0]+0.5*dt*dq[0], s.q[1]+0.5*dt*dq[1], s.q[2]+0.5*dt*dq[2], s.q[3]+0.5*dt*dq[3]];
    let nrm = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
    for k in 0..4 { q[k] /= nrm; }
    s.q = q;
    if s.p[2] < p.radius { s.done = true; }
}

// ---------------- SHA-256 (deps-free) ----------------
const K256: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let ml = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&ml.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[4*i], chunk[4*i+1], chunk[4*i+2], chunk[4*i+3]]); }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K256[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for i in 0..8 { out[4*i..4*i+4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}
fn hexstr(d: &[u8; 32]) -> String { d.iter().map(|b| format!("{:02x}", b)).collect() }

fn main() {
    // known-answer test for the SHA-256 implementation
    assert_eq!(hexstr(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(hexstr(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    println!("SHA-256 self-test: PASS");

    // frsqrt precision across the system's range
    let test_x = [1.0f64, 0.99, 1.01, 0.5, 2.0, 100.0, 7500.0, 1e-4, 1e-6, 0.0001525];
    let mut worst = 0.0f64;
    for &x in &test_x {
        let got = dbl(frsqrt(fx(x)));
        let want = 1.0 / x.sqrt();
        let rel = ((got - want) / want).abs();
        if rel > worst { worst = rel; }
    }
    println!("frsqrt worst relative error over range = {:.3e}  (Q32.32 quantization floor)", worst);
    assert!(worst < 1e-5, "frsqrt not precise enough");

    // ---- params ----
    let l = 0.17f64;
    let a = l / std::f64::consts::SQRT_2;
    let pf = Pf { m: 0.5, g: 9.81, ix: 3.2e-3, iy: 3.2e-3, iz: 5.5e-3, kf: 1.0e-6, km: 1.6e-8, tau: 0.02, cd: 0.1, radius: 0.10,
        rpos: [(a, a), (-a, -a), (a, -a), (-a, a)], spin: [1.0, 1.0, -1.0, -1.0] };
    let fp = FxP {
        kf: fx(pf.kf), km: fx(pf.km), cd: fx(pf.cd), g: fx(pf.g), radius: fx(pf.radius), dt: fx(1e-3), half_dt: fx(0.5e-3),
        ix: fx(pf.ix), iy: fx(pf.iy), iz: fx(pf.iz),
        inv_tau: fx(1.0/pf.tau), inv_m: fx(1.0/pf.m), inv_ix: fx(1.0/pf.ix), inv_iy: fx(1.0/pf.iy), inv_iz: fx(1.0/pf.iz),
        rpos: [(fx(a), fx(a)), (fx(-a), fx(-a)), (fx(a), fx(-a)), (fx(-a), fx(a))],
        spin: [fx(1.0), fx(1.0), fx(-1.0), fx(-1.0)],
    };
    let oh = ((pf.m*pf.g/4.0)/pf.kf).sqrt();
    let d = 30.0;
    let cmd_f = [oh*1.2+d, oh*1.2-d, oh*1.2-d, oh*1.2+d];
    let cmd_x = [fx(cmd_f[0]), fx(cmd_f[1]), fx(cmd_f[2]), fx(cmd_f[3])];
    let dt = 1e-3;
    let n = 1500;

    // ---- rollouts ----
    let mut sf = Sf { p: [0.0, 0.0, 10.0], v: [0.0; 3], q: [0.0, 0.0, 0.0, 1.0], w: [0.0; 3], r: [oh; 4], done: false };
    let mut sx = FxS { p: [0, 0, fx(10.0)], v: [0; 3], q: [0, 0, 0, ONE], w: [0; 3], r: [fx(oh); 4], done: false };

    let mut traj: Vec<u8> = Vec::with_capacity(n * 18 * 8);
    for _ in 0..n {
        fstep_f64(&mut sf, &pf, cmd_f, dt);
        fstep(&mut sx, &fp, cmd_x);
        // append fixed-point state (18 x i64, little-endian) to the trajectory
        let words = [sx.p[0], sx.p[1], sx.p[2], sx.v[0], sx.v[1], sx.v[2],
                     sx.q[0], sx.q[1], sx.q[2], sx.q[3], sx.w[0], sx.w[1], sx.w[2],
                     sx.r[0], sx.r[1], sx.r[2], sx.r[3], sx.done as i64];
        for wv in words { traj.extend_from_slice(&wv.to_le_bytes()); }
    }

    let gp = [dbl(sx.p[0]), dbl(sx.p[1]), dbl(sx.p[2])];
    let gv = [dbl(sx.v[0]), dbl(sx.v[1]), dbl(sx.v[2])];
    let dp = ((sf.p[0]-gp[0]).powi(2)+(sf.p[1]-gp[1]).powi(2)+(sf.p[2]-gp[2]).powi(2)).sqrt();
    let dv = ((sf.v[0]-gv[0]).powi(2)+(sf.v[1]-gv[1]).powi(2)+(sf.v[2]-gv[2]).powi(2)).sqrt();
    let qf = sf.q;
    let qx = [dbl(sx.q[0]), dbl(sx.q[1]), dbl(sx.q[2]), dbl(sx.q[3])];
    let dotq = (qf[0]*qx[0]+qf[1]*qx[1]+qf[2]*qx[2]+qf[3]*qx[3]).abs().min(1.0);
    let ang = 2.0*dotq.acos()*180.0/std::f64::consts::PI;

    println!("\nscenario: 1.5 s, +20% collective + roll differential, {} substeps", n);
    println!("f64 reference final pos = ({:.4}, {:.4}, {:.4})", sf.p[0], sf.p[1], sf.p[2]);
    println!("Q32.32 fixed  final pos = ({:.4}, {:.4}, {:.4})", gp[0], gp[1], gp[2]);
    println!("fixed-vs-f64 divergence (quantization of coefficients + arithmetic):");
    println!("  position = {:.3e} m", dp);
    println!("  velocity = {:.3e} m/s", dv);
    println!("  attitude = {:.3e} deg", ang);

    let digest = sha256(&traj);
    println!("\ntrajectory bytes = {}  ({} substeps x 18 x i64)", traj.len(), n);
    println!("SHA-256(fixed-point trajectory) = {}", hexstr(&digest));

    // determinism: recompute and confirm identical hash
    let mut sx2 = FxS { p: [0, 0, fx(10.0)], v: [0; 3], q: [0, 0, 0, ONE], w: [0; 3], r: [fx(oh); 4], done: false };
    let mut traj2: Vec<u8> = Vec::with_capacity(traj.len());
    for _ in 0..n {
        fstep(&mut sx2, &fp, cmd_x);
        let words = [sx2.p[0], sx2.p[1], sx2.p[2], sx2.v[0], sx2.v[1], sx2.v[2],
                     sx2.q[0], sx2.q[1], sx2.q[2], sx2.q[3], sx2.w[0], sx2.w[1], sx2.w[2],
                     sx2.r[0], sx2.r[1], sx2.r[2], sx2.r[3], sx2.done as i64];
        for wv in words { traj2.extend_from_slice(&wv.to_le_bytes()); }
    }
    assert_eq!(digest, sha256(&traj2), "run-to-run determinism failed");
    assert!(dp < 0.02 && ang < 0.5, "fixed-point drifted too far from reference");
    println!("run-to-run determinism: PASS (identical SHA-256)");
    println!("\nfixed-point step VERIFIED against f64 reference. This is the exact algorithm the WGSL kernel mirrors.");
}

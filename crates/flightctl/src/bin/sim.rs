//! Sim-side host for the control seam. Closes the loop: physics solver produces
//! state -> we marshal it across the SAME `control()` ABI -> the returned rotor
//! commands drive `step`. This is the browser-sim shim; the MAVLink shim (sketched
//! at the bottom) wraps the identical `control()` call with different I/O.

use flightctl::control;

// f64 plant (the M1 reference dynamics; quaternion (x,y,z,w))
struct Plant { p: [f64; 3], v: [f64; 3], q: [f64; 4], w: [f64; 3], r: [f64; 4] }
const M: f64 = 0.5; const G: f64 = 9.81;
const KF: f64 = 1.0e-6; const KM: f64 = 1.6e-8; const TAU: f64 = 0.02; const CD: f64 = 0.1;
const IX: f64 = 3.2e-3; const IY: f64 = 3.2e-3; const IZ: f64 = 5.5e-3;
const ARM: f64 = 0.120208;
const RPOS: [(f64, f64); 4] = [(ARM, ARM), (-ARM, -ARM), (ARM, -ARM), (-ARM, ARM)];
const SPIN: [f64; 4] = [1.0, 1.0, -1.0, -1.0];

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]]; let s = q[3];
    let c1 = cross(u, v); let t = [2.0*c1[0], 2.0*c1[1], 2.0*c1[2]]; let c2 = cross(u, t);
    [v[0]+s*t[0]+c2[0], v[1]+s*t[1]+c2[1], v[2]+s*t[2]+c2[2]]
}
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let av = [a[0], a[1], a[2]]; let aw = a[3]; let bv = [b[0], b[1], b[2]]; let bw = b[3];
    let c = cross(av, bv);
    [aw*bv[0]+bw*av[0]+c[0], aw*bv[1]+bw*av[1]+c[1], aw*bv[2]+bw*av[2]+c[2],
     aw*bw-(av[0]*bv[0]+av[1]*bv[1]+av[2]*bv[2])]
}
fn step(s: &mut Plant, cmd: [f64; 4], dt: f64) {
    for i in 0..4 { s.r[i] += dt*(cmd[i]-s.r[i])/TAU; }
    let mut th = [0.0; 4]; let mut re = [0.0; 4];
    for i in 0..4 { let r2 = s.r[i]*s.r[i]; th[i]=KF*r2; re[i]=KM*r2; }
    let fz = th[0]+th[1]+th[2]+th[3];
    let tx = RPOS[0].1*th[0]+RPOS[1].1*th[1]+RPOS[2].1*th[2]+RPOS[3].1*th[3];
    let ty = -(RPOS[0].0*th[0]+RPOS[1].0*th[1]+RPOS[2].0*th[2]+RPOS[3].0*th[3]);
    let tz = SPIN[0]*re[0]+SPIN[1]*re[1]+SPIN[2]*re[2]+SPIN[3]*re[3];
    let tw = qrot(s.q, [0.0, 0.0, fz]);
    let sp = (s.v[0]*s.v[0]+s.v[1]*s.v[1]+s.v[2]*s.v[2]).sqrt();
    let acc = [(tw[0]-CD*sp*s.v[0])/M, (tw[1]-CD*sp*s.v[1])/M, (tw[2]-CD*sp*s.v[2])/M - G];
    for k in 0..3 { s.v[k] += dt*acc[k]; }
    for k in 0..3 { s.p[k] += dt*s.v[k]; }
    let iw = [IX*s.w[0], IY*s.w[1], IZ*s.w[2]];
    let gyro = cross(s.w, iw);
    let wd = [(tx-gyro[0])/IX, (ty-gyro[1])/IY, (tz-gyro[2])/IZ];
    for k in 0..3 { s.w[k] += dt*wd[k]; }
    let wq = [s.w[0], s.w[1], s.w[2], 0.0];
    let dq = qmul(s.q, wq);
    let mut q = [s.q[0]+0.5*dt*dq[0], s.q[1]+0.5*dt*dq[1], s.q[2]+0.5*dt*dq[2], s.q[3]+0.5*dt*dq[3]];
    let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
    for k in 0..4 { q[k] /= n; }
    s.q = q;
}

fn main() {
    let oh = ((M*G/4.0)/KF).sqrt();
    let mut plant = Plant { p: [0.0, 0.0, 10.0], v: [0.0; 3], q: [0.0, 0.0, 0.0, 1.0], w: [0.0; 3], r: [oh; 4] };
    let setpoint: [f32; 4] = [2.0, -1.0, 12.0, 0.0]; // fly to (2,-1,12), hold

    let dt = 1e-3;
    let secs = 8.0;
    let n = (secs / dt) as usize;
    println!("closed loop: start (0,0,10) -> setpoint ({}, {}, {}), {} s @ {} kHz control",
             setpoint[0], setpoint[1], setpoint[2], secs, 1.0/dt/1000.0);
    println!("   t(s)      x        y        z       |v|      tilt(deg)");

    let mut out = [0f32; 4];
    for k in 0..n {
        // ---- sim shim: marshal plant state across the control ABI ----
        let st: [f32; 17] = [
            plant.p[0] as f32, plant.p[1] as f32, plant.p[2] as f32,
            plant.v[0] as f32, plant.v[1] as f32, plant.v[2] as f32,
            plant.q[0] as f32, plant.q[1] as f32, plant.q[2] as f32, plant.q[3] as f32,
            plant.w[0] as f32, plant.w[1] as f32, plant.w[2] as f32,
            plant.r[0] as f32, plant.r[1] as f32, plant.r[2] as f32, plant.r[3] as f32,
        ];
        control(st.as_ptr(), setpoint.as_ptr(), out.as_mut_ptr());
        let cmd = [out[0] as f64, out[1] as f64, out[2] as f64, out[3] as f64];
        // ---- end shim ----
        step(&mut plant, cmd, dt);

        if k % 1000 == 0 || k == n - 1 {
            let t = (k + 1) as f64 * dt;
            let spd = (plant.v[0].powi(2) + plant.v[1].powi(2) + plant.v[2].powi(2)).sqrt();
            let zcur = qrot(plant.q, [0.0, 0.0, 1.0]);
            let tilt = zcur[2].clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI;
            println!("  {:5.2}  {:7.3}  {:7.3}  {:7.3}  {:7.3}   {:6.2}", t, plant.p[0], plant.p[1], plant.p[2], spd, tilt);
        }
    }

    let err = ((plant.p[0]-2.0).powi(2) + (plant.p[1]+1.0).powi(2) + (plant.p[2]-12.0).powi(2)).sqrt();
    let spd = (plant.v[0].powi(2)+plant.v[1].powi(2)+plant.v[2].powi(2)).sqrt();
    println!("\nfinal position error = {:.4} m, final speed = {:.4} m/s", err, spd);
    assert!(err < 0.02 && spd < 0.05, "controller did not converge to setpoint");
    println!("CONVERGED: the same control() that ships as .wasm flew the drone to the setpoint in sim.");
}

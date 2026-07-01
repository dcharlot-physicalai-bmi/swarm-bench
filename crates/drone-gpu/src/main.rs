//! Milestones 2–3 — batched GPU quadrotor solver.
//!
//! Runs N independent environments through the WGSL kernel (drone_step.wgsl) in a
//! single compute dispatch (each env loops n_substeps internally). Verifies:
//!   (A) per-env independence  — N identical envs produce byte-identical results,
//!       and env 0 matches an f64 reference within f32 tolerance;
//!   (B) heterogeneous batching — N envs with distinct commands/altitudes, spot
//!       checked against per-env f64 references (proves field-major indexing);
//!   (C) scaling — envs/sec and env-substeps/sec across N.
//! Uses wgpu DEFAULT limits (WebGPU baseline): 3 storage buffers, field-major state.

use std::time::Instant;
use wgpu::util::DeviceExt;

// ------------------------------ f64 reference ------------------------------
#[derive(Clone, Copy)]
struct P {
    m: f64, g: f64, ix: f64, iy: f64, iz: f64,
    kf: f64, km: f64, tau_m: f64, cd: f64, radius: f64,
    rpos: [(f64, f64); 4], spin: [f64; 4],
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let (av, aw) = ([a[0], a[1], a[2]], a[3]);
    let (bv, bw) = ([b[0], b[1], b[2]], b[3]);
    let c = cross(av, bv);
    [aw*bv[0]+bw*av[0]+c[0], aw*bv[1]+bw*av[1]+c[1], aw*bv[2]+bw*av[2]+c[2],
     aw*bw-(av[0]*bv[0]+av[1]*bv[1]+av[2]*bv[2])]
}
fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]]; let s = q[3];
    let c1 = cross(u, v);
    let t = [2.0*c1[0], 2.0*c1[1], 2.0*c1[2]];
    let c2 = cross(u, t);
    [v[0]+s*t[0]+c2[0], v[1]+s*t[1]+c2[1], v[2]+s*t[2]+c2[2]]
}
#[derive(Clone, Copy)]
struct RefState { p: [f64; 3], v: [f64; 3], q: [f64; 4], w: [f64; 3], r: [f64; 4], done: bool }
fn ref_step(s: &mut RefState, p: &P, c: [f64; 4], dt: f64) {
    if s.done { return; }
    for i in 0..4 { s.r[i] += dt*(c[i]-s.r[i])/p.tau_m; }
    let th = [p.kf*s.r[0]*s.r[0], p.kf*s.r[1]*s.r[1], p.kf*s.r[2]*s.r[2], p.kf*s.r[3]*s.r[3]];
    let qd = [p.km*s.r[0]*s.r[0], p.km*s.r[1]*s.r[1], p.km*s.r[2]*s.r[2], p.km*s.r[3]*s.r[3]];
    let fz = th[0]+th[1]+th[2]+th[3];
    let tx = p.rpos[0].1*th[0]+p.rpos[1].1*th[1]+p.rpos[2].1*th[2]+p.rpos[3].1*th[3];
    let ty = -(p.rpos[0].0*th[0]+p.rpos[1].0*th[1]+p.rpos[2].0*th[2]+p.rpos[3].0*th[3]);
    let tz = p.spin[0]*qd[0]+p.spin[1]*qd[1]+p.spin[2]*qd[2]+p.spin[3]*qd[3];
    let tau = [tx, ty, tz];
    let tw = qrot(s.q, [0.0, 0.0, fz]);
    let speed = (s.v[0]*s.v[0]+s.v[1]*s.v[1]+s.v[2]*s.v[2]).sqrt();
    let accel = [(tw[0]-p.cd*speed*s.v[0])/p.m, (tw[1]-p.cd*speed*s.v[1])/p.m, (tw[2]-p.cd*speed*s.v[2])/p.m - p.g];
    for k in 0..3 { s.v[k] += dt*accel[k]; }
    for k in 0..3 { s.p[k] += dt*s.v[k]; }
    let iw = [p.ix*s.w[0], p.iy*s.w[1], p.iz*s.w[2]];
    let gyro = cross(s.w, iw);
    let wdot = [(tau[0]-gyro[0])/p.ix, (tau[1]-gyro[1])/p.iy, (tau[2]-gyro[2])/p.iz];
    for k in 0..3 { s.w[k] += dt*wdot[k]; }
    let wq = [s.w[0], s.w[1], s.w[2], 0.0];
    let dq = qmul(s.q, wq);
    let mut q = [s.q[0]+0.5*dt*dq[0], s.q[1]+0.5*dt*dq[1], s.q[2]+0.5*dt*dq[2], s.q[3]+0.5*dt*dq[3]];
    let nrm = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
    for k in 0..4 { q[k] /= nrm; }
    s.q = q;
    if s.p[2] < p.radius { s.done = true; }
}
fn ref_rollout(p: &P, z0: f64, cmd: [f64; 4], oh: f64, dt: f64, n: u32) -> RefState {
    let mut s = RefState { p: [0.0, 0.0, z0], v: [0.0; 3], q: [0.0, 0.0, 0.0, 1.0], w: [0.0; 3], r: [oh; 4], done: false };
    for _ in 0..n { ref_step(&mut s, p, cmd, dt); }
    s
}

// ------------------------------ GPU config ------------------------------
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuConfig {
    m: f32, g: f32, ix: f32, iy: f32, iz: f32,
    kf: f32, km: f32, tau_m: f32, cd: f32, radius: f32,
    rx0: f32, ry0: f32, rx1: f32, ry1: f32, rx2: f32, ry2: f32, rx3: f32, ry3: f32,
    s0: f32, s1: f32, s2: f32, s3: f32,
    dt: f32, n_env: u32, n_substeps: u32, pad: u32,
}
fn make_config(p: &P, dt: f64, n_env: u32, n_substeps: u32) -> GpuConfig {
    GpuConfig {
        m: p.m as f32, g: p.g as f32, ix: p.ix as f32, iy: p.iy as f32, iz: p.iz as f32,
        kf: p.kf as f32, km: p.km as f32, tau_m: p.tau_m as f32, cd: p.cd as f32, radius: p.radius as f32,
        rx0: p.rpos[0].0 as f32, ry0: p.rpos[0].1 as f32, rx1: p.rpos[1].0 as f32, ry1: p.rpos[1].1 as f32,
        rx2: p.rpos[2].0 as f32, ry2: p.rpos[2].1 as f32, rx3: p.rpos[3].0 as f32, ry3: p.rpos[3].1 as f32,
        s0: p.spin[0] as f32, s1: p.spin[1] as f32, s2: p.spin[2] as f32, s3: p.spin[3] as f32,
        dt: dt as f32, n_env, n_substeps, pad: 0,
    }
}

// field-major builders (value of field f for env i at index f*n + i)
fn build_state<F: Fn(usize) -> (f32, [f32; 4])>(n: usize, oh: f32, gen: &F) -> Vec<f32> {
    let mut s = vec![0.0f32; 18 * n];
    for i in 0..n {
        let (z, _c) = gen(i);
        s[2 * n + i] = z;         // pz
        s[9 * n + i] = 1.0;       // qw (identity)
        s[13 * n + i] = oh;       // rotor 0..3
        s[14 * n + i] = oh;
        s[15 * n + i] = oh;
        s[16 * n + i] = oh;
    }
    s
}
fn build_cmd<F: Fn(usize) -> (f32, [f32; 4])>(n: usize, gen: &F) -> Vec<f32> {
    let mut c = vec![0.0f32; 4 * n];
    for i in 0..n {
        let (_z, cmd) = gen(i);
        for j in 0..4 { c[j * n + i] = cmd[j]; }
    }
    c
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None, force_fallback_adapter: false, compatible_surface: None,
        })).expect("no Vulkan adapter");
        let info = adapter.get_info();
        println!("GPU adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
        let lim = adapter.limits();
        println!("adapter max_storage_buffer_binding_size = {} MiB", lim.max_storage_buffer_binding_size / (1024*1024));
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None, required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default(),
        }, None)).expect("request_device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drone_step"), source: wgpu::ShaderSource::Wgsl(include_str!("drone_step.wgsl").into()),
        });
        let entry = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[entry(0, false), entry(1, true), entry(2, true)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &shader, entry_point: "step" });
        Gpu { device, queue, pipeline, bgl }
    }

    // dispatch; returns (final state Vec<f32> length 18*n, compute elapsed seconds)
    fn run<F: Fn(usize) -> (f32, [f32; 4])>(&self, p: &P, oh: f32, dt: f64, n: usize, n_sub: u32, gen: &F) -> (Vec<f32>, f64) {
        let state = build_state(n, oh, gen);
        let cmd = build_cmd(n, gen);
        let cfg = make_config(p, dt, n as u32, n_sub);
        let state_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("state"), contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let cmd_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cmd"), contents: bytemuck::cast_slice(&cmd), usage: wgpu::BufferUsages::STORAGE,
        });
        let cfg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cfg"), contents: bytemuck::bytes_of(&cfg), usage: wgpu::BufferUsages::STORAGE,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cmd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
            ],
        });
        let groups = ((n as u32) + 63) / 64;
        let t0 = Instant::now();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed = t0.elapsed().as_secs_f64();

        // readback
        let size = (18 * n * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut enc2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc2.copy_buffer_to_buffer(&state_buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc2.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        (out, elapsed)
    }
}

fn get(state: &[f32], n: usize, field: usize, i: usize) -> f64 { state[field * n + i] as f64 }

fn main() {
    let l = 0.17f64;
    let a = l / std::f64::consts::SQRT_2;
    let p = P {
        m: 0.5, g: 9.81, ix: 3.2e-3, iy: 3.2e-3, iz: 5.5e-3,
        kf: 1.0e-6, km: 1.6e-8, tau_m: 0.02, cd: 0.1, radius: 0.10,
        rpos: [(a, a), (-a, -a), (a, -a), (-a, a)], spin: [1.0, 1.0, -1.0, -1.0],
    };
    let oh = ((p.m * p.g / 4.0) / p.kf).sqrt();
    let ohf = oh as f32;
    let dt = 1e-3;

    let gpu = Gpu::new();

    // heterogeneous env generator: distinct collective, differential, altitude per env
    let gen_var = |i: usize| -> (f32, [f32; 4]) {
        let cf = 1.1 + 0.2 * ((i % 7) as f32) / 6.0;
        let d = 10.0 + 5.0 * (i % 5) as f32;
        let z = 10.0 + (i % 11) as f32;
        let base = ohf * cf;
        (z, [base + d, base - d, base - d, base + d])
    };
    // identical env generator (the milestone-2 scenario)
    let gen_same = |_i: usize| -> (f32, [f32; 4]) {
        let base = ohf * 1.2;
        (10.0, [base + 30.0, base - 30.0, base - 30.0, base + 30.0])
    };

    // ---------- (A) independence: N identical envs ----------
    let n_a = 4096usize;
    let n_sub = 1500u32;
    let (sa, _t) = gpu.run(&p, ohf, dt, n_a, n_sub, &gen_same);
    // max deviation of any field of any env from env 0
    let mut max_dev = 0.0f64;
    for f in 0..18 {
        let ref0 = sa[f * n_a + 0];
        for i in 0..n_a {
            let dv = (sa[f * n_a + i] - ref0).abs() as f64;
            if dv > max_dev { max_dev = dv; }
        }
    }
    // env 0 vs f64 reference
    let (z0, cmd0) = gen_same(0);
    let r0 = ref_rollout(&p, z0 as f64, [cmd0[0] as f64, cmd0[1] as f64, cmd0[2] as f64, cmd0[3] as f64], ohf as f64, dt, n_sub);
    let dp0 = ((r0.p[0]-get(&sa,n_a,0,0)).powi(2)+(r0.p[1]-get(&sa,n_a,1,0)).powi(2)+(r0.p[2]-get(&sa,n_a,2,0)).powi(2)).sqrt();
    println!("\n(A) independence — {} identical envs, {} substeps:", n_a, n_sub);
    println!("    max inter-env field deviation = {:.3e}  (expect exactly 0)", max_dev);
    println!("    env 0 position vs f64 reference = {:.3e} m", dp0);
    assert!(max_dev == 0.0, "envs not independent/identical");
    assert!(dp0 < 0.05);

    // ---------- (B) heterogeneous batching ----------
    let n_b = 4096usize;
    let (sb, _t) = gpu.run(&p, ohf, dt, n_b, n_sub, &gen_var);
    println!("\n(B) heterogeneous batching — {} distinct envs, {} substeps:", n_b, n_sub);
    let mut worst = 0.0f64;
    for &i in &[0usize, 1, 777, 2048, n_b - 1] {
        let (z, cmd) = gen_var(i);
        let r = ref_rollout(&p, z as f64, [cmd[0] as f64, cmd[1] as f64, cmd[2] as f64, cmd[3] as f64], ohf as f64, dt, n_sub);
        let dp = ((r.p[0]-get(&sb,n_b,0,i)).powi(2)+(r.p[1]-get(&sb,n_b,1,i)).powi(2)+(r.p[2]-get(&sb,n_b,2,i)).powi(2)).sqrt();
        let dotq = (r.q[0]*get(&sb,n_b,6,i)+r.q[1]*get(&sb,n_b,7,i)+r.q[2]*get(&sb,n_b,8,i)+r.q[3]*get(&sb,n_b,9,i)).abs().min(1.0);
        let ang = 2.0*dotq.acos()*180.0/std::f64::consts::PI;
        println!("    env {:>5}: pos Δ = {:.3e} m, attitude Δ = {:.3e} deg", i, dp, ang);
        if dp > worst { worst = dp; }
    }
    assert!(worst < 0.05, "heterogeneous env diverged");

    // ---------- (C) scaling ----------
    let sweep = [1usize, 64, 1024, 16384, 65536, 262144];
    let n_sub_s = 128u32;
    println!("\n(C) scaling — n_substeps = {} per env:", n_sub_s);
    println!("    {:>9}  {:>12}  {:>16}  {:>16}", "N_env", "compute (ms)", "envs/sec", "env-substeps/sec");
    for &n in &sweep {
        let (_s, t) = gpu.run(&p, ohf, dt, n, n_sub_s, &gen_var);
        let eps = n as f64 / t;
        let ess = (n as f64 * n_sub_s as f64) / t;
        println!("    {:>9}  {:>12.3}  {:>16.3e}  {:>16.3e}", n, t * 1e3, eps, ess);
    }
    println!("\n(note: adapter above is llvmpipe — CPU software Vulkan. Absolute rates are a");
    println!(" CPU floor; on real GPU silicon throughput is orders of magnitude higher. The");
    println!(" point here is correctness under batching and that the harness scales cleanly.)");
    println!("\nALL BATCHED CHECKS PASSED.");
}

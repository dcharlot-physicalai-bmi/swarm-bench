//! Milestone 4 keystone — verify exact Q32.32 fixed-point multiply on GPU.
//!
//! Runs the emulated multiply (mul_q32.wgsl) on a batch of operand pairs spanning
//! the quadrotor system's magnitudes, then checks each GPU result BIT-EXACTLY
//! against an i128 reference. Bit-exact here + integer-only ops in the shader is
//! what makes the eventual fixed-point kernel reproducible across devices.

use wgpu::util::DeviceExt;

const SCALE: f64 = 4294967296.0; // 2^32

fn to_q32(x: f64) -> i64 {
    (x * SCALE).round() as i64
}
fn from_q32(v: i64) -> f64 {
    v as f64 / SCALE
}
fn limbs(v: i64) -> (u32, u32) {
    let u = v as u64;
    (u as u32, (u >> 32) as u32)
}
fn unlimbs(lo: u32, hi: u32) -> i64 {
    ((((hi as u64) << 32) | lo as u64) as u64) as i64
}
// exact fixed-point product reference. Truncate toward zero (i128 division),
// matching the WGSL magnitude-then-sign path. This is the kernel's fixed
// rounding convention: unambiguous and identical on every device.
fn ref_mul(a: i64, b: i64) -> i64 {
    ((a as i128) * (b as i128) / (1i128 << 32)) as i64
}

fn main() {
    // operand pairs (real values) spanning the dynamics
    let pairs: &[(f64, f64)] = &[
        (1107.0, 1107.0),          // Ω * Ω  (≈1.2e6, overflows Q16.16, fits Q32.32)
        (1.0e-6, 1_225_449.0),     // kf * Ω²  -> thrust ≈ 1.225 N
        (1.6e-8, 1_225_449.0),     // km * Ω²  -> reaction torque
        (0.001, 9810.0),           // dt * (a·something)
        (0.70710678, -0.70710678), // quaternion components, mixed sign
        (312.5, 0.0012),           // 1/Ixx * torque
        (-1500.0, 1500.0),         // near max rotor, opposite signs (≈-2.25e6)
        (2.5, -3.7),
        (1.52e-5, 65535.9),        // tiny * large
        (1.0e-4, 1.0e-4),          // small * small (≈1e-8, near resolution)
        (-1.0, 1.0),
        (0.12013, 0.12013),        // arm coord a=0.1202 squared
    ];

    // pack input: 4 u32 per pair (a_lo,a_hi,b_lo,b_hi)
    let mut inp: Vec<u32> = Vec::with_capacity(pairs.len() * 4);
    let mut a_fixed = Vec::new();
    let mut b_fixed = Vec::new();
    for &(a, b) in pairs {
        let af = to_q32(a);
        let bf = to_q32(b);
        a_fixed.push(af);
        b_fixed.push(bf);
        let (al, ah) = limbs(af);
        let (bl, bh) = limbs(bf);
        inp.extend_from_slice(&[al, ah, bl, bh]);
    }
    let count = pairs.len() as u32;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Cfg { count: u32, p0: u32, p1: u32, p2: u32 }
    let cfg = Cfg { count, p0: 0, p1: 0, p2: 0 };

    // ---- GPU ----
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None, force_fallback_adapter: false, compatible_surface: None,
    })).expect("no adapter");
    println!("GPU adapter: {} ({:?}, {:?})", adapter.get_info().name, adapter.get_info().device_type, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None, required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default(),
    }, None)).expect("device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mul_q32"), source: wgpu::ShaderSource::Wgsl(include_str!("../mul_q32.wgsl").into()),
    });

    let inp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("inp"), contents: bytemuck::cast_slice(&inp), usage: wgpu::BufferUsages::STORAGE,
    });
    let out_len = (count as usize) * 2;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("outp"), size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let cfg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cfg"), contents: bytemuck::bytes_of(&cfg), usage: wgpu::BufferUsages::STORAGE,
    });

    let entry = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[entry(0, true), entry(1, false), entry(2, true)],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cfg_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &shader, entry_point: "run" });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((count + 63) / 64, 1, 1);
    }
    queue.submit(Some(enc.finish()));

    // readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (out_len * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (out_len * 4) as u64);
    queue.submit(Some(enc2.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    // ---- compare bit-exactly to i128 reference ----
    println!("\n{:>14} {:>14} {:>18} {:>12}", "a", "b", "a*b (Q32.32)", "vs i128");
    let mut all_ok = true;
    let mut worst_real_err = 0.0f64;
    for i in 0..pairs.len() {
        let gpu = unlimbs(out[2 * i], out[2 * i + 1]);
        let exp = ref_mul(a_fixed[i], b_fixed[i]);
        let ok = gpu == exp;
        all_ok &= ok;
        let gpu_real = from_q32(gpu);
        let true_real = pairs[i].0 * pairs[i].1;
        let err = (gpu_real - true_real).abs();
        if err > worst_real_err { worst_real_err = err; }
        println!("{:>14.6} {:>14.6} {:>18.6} {:>12}", pairs[i].0, pairs[i].1, gpu_real, if ok { "exact" } else { "MISMATCH" });
    }
    println!("\nworst error vs true real product = {:.3e}  (Q32.32 resolution = {:.3e})", worst_real_err, 1.0 / SCALE);
    assert!(all_ok, "GPU fixed-point multiply did not match i128 reference");
    println!("PASS: emulated Q32.32 multiply is bit-exact to the i128 reference on this device.");
    println!("      integer-only + bit-exact => reproducible across devices (the determinism basis).");
}

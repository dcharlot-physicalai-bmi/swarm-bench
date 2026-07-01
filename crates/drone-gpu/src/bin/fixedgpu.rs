//! Milestone 4 — GPU determinism proof.
//!
//! Runs the full Q32.32 fixed-point step on the GPU (fixed_step.wgsl) and checks
//! its per-substep trajectory BYTE-FOR-BYTE against the native-i64 CPU reference
//! (fixed_ref). Two independent integer implementations on two substrates (CPU
//! i64/i128, GPU emulated-64-in-WGSL) agreeing bit-for-bit is the determinism
//! result; the same SHA-256 is the regression anchor to re-check on other GPUs.

use wgpu::util::DeviceExt;

#[path = "../fixed_ref.rs"]
mod fixed_ref;

fn main() {
    // SHA-256 self-test
    assert_eq!(fixed_ref::hexstr(&fixed_ref::sha256(b"abc")),
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    let sc = fixed_ref::scenario();
    let cpu_digest = fixed_ref::sha256(&sc.traj);
    println!("scenario: 1.5 s, {} substeps, Q32.32 fixed-point", sc.nsub);
    println!("CPU (native i64) final pos = ({:.4}, {:.4}, {:.4})", sc.final_pos[0], sc.final_pos[1], sc.final_pos[2]);

    // ---- GPU ----
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None, force_fallback_adapter: false, compatible_surface: None,
    })).expect("no adapter");
    println!("GPU adapter: {} ({:?}, {:?})", adapter.get_info().name, adapter.get_info().device_type, adapter.get_info().backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None, required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default(),
    }, None)).expect("device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fixed_step"), source: wgpu::ShaderSource::Wgsl(include_str!("../fixed_step.wgsl").into()),
    });

    let mk = |data: &[u32], rw: bool| -> wgpu::Buffer {
        let usage = if rw { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST }
                    else { wgpu::BufferUsages::STORAGE };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage })
    };
    let st_buf = mk(&sc.state, false);
    let cmd_buf = mk(&sc.cmd, false);
    let prm_buf = mk(&sc.prm, false);
    let meta_buf = mk(&[1u32, sc.nsub], false);
    let traj_len = (sc.nsub as usize) * 36;
    let traj_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("traj"), size: (traj_len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });

    let entry = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None, entries: &[entry(0, true), entry(1, true), entry(2, true), entry(3, true), entry(4, false)],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: st_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cmd_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: prm_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: traj_buf.as_entire_binding() },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pl), module: &shader, entry_point: "run" });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(Some(enc.finish()));

    // readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (traj_len * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_buffer_to_buffer(&traj_buf, 0, &staging, 0, (traj_len * 4) as u64);
    queue.submit(Some(enc2.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let gpu_bytes: Vec<u8> = data.to_vec(); // little-endian u32 stream == i64 LE stream
    drop(data);
    staging.unmap();

    // ---- compare ----
    let gpu_digest = fixed_ref::sha256(&gpu_bytes);
    let bytes_match = gpu_bytes == sc.traj;
    let first_diff = sc.traj.iter().zip(gpu_bytes.iter()).position(|(a, b)| a != b);

    println!("\ntrajectory bytes: CPU {}, GPU {}", sc.traj.len(), gpu_bytes.len());
    println!("byte-for-byte identical: {}", bytes_match);
    if let Some(i) = first_diff { println!("  first differing byte at offset {}", i); }
    println!("\nSHA-256(CPU i64 trajectory) = {}", fixed_ref::hexstr(&cpu_digest));
    println!("SHA-256(GPU emulated-i64 trajectory) = {}", fixed_ref::hexstr(&gpu_digest));

    assert!(bytes_match && cpu_digest == gpu_digest, "GPU fixed-point trajectory differs from CPU reference");
    println!("\nPASS: CPU native-i64 and GPU emulated-64-bit produce a BIT-IDENTICAL trajectory.");
    println!("Store this SHA-256 as the cross-device regression anchor; re-run on any other GPU");
    println!("and an identical digest proves bit-exact reproducibility across that hardware.");
}

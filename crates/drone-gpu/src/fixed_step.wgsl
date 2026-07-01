// Full Q32.32 fixed-point quadrotor step in WGSL. Integer-only: 64-bit values are
// carried as two u32 limbs (lo,hi); multiply builds a 128-bit product; inverse-sqrt
// is division-free Newton seeded by bit position. This is a 1:1 transcription of the
// Rust fixed_ref::fstep, so the two implementations agree bit-for-bit. N = 1; the
// kernel writes the full per-substep trajectory for hashing.

alias fx = vec2<u32>;                 // .x = lo, .y = hi (two's-complement 64-bit)
alias v3 = array<fx, 3>;
alias v4 = array<fx, 4>;

const ZERO         = fx(0u, 0u);
const F_HALF       = fx(0x80000000u, 0u);   // 0.5
const F_THREE_HALF = fx(0x80000000u, 1u);   // 1.5
const F_TWO        = fx(0u, 2u);            // 2.0

@group(0) @binding(0) var<storage, read>       st:   array<u32>; // 36 u32 : initial state
@group(0) @binding(1) var<storage, read>       cmdb: array<u32>; // 8 u32
@group(0) @binding(2) var<storage, read>       prm:  array<u32>; // 54 u32
@group(0) @binding(3) var<storage, read>       dims: array<u32>; // [n_env, n_substeps]
@group(0) @binding(4) var<storage, read_write> traj: array<u32>; // n_substeps * 36 u32

fn P(i: u32) -> fx { return fx(prm[2u*i], prm[2u*i+1u]); }
fn getw(i: u32) -> fx { return fx(st[2u*i], st[2u*i+1u]); }
fn getc(i: u32) -> fx { return fx(cmdb[2u*i], cmdb[2u*i+1u]); }
fn putw(k: u32, wi: u32, val: fx) { let b = k*36u + 2u*wi; traj[b] = val.x; traj[b+1u] = val.y; }

// ---- 64-bit integer / Q32.32 primitives ----
fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b; let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin; let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 | c2);
}
fn fadd(a: fx, b: fx) -> fx {
    let lo = a.x + b.x; let c = select(0u, 1u, lo < a.x);
    return fx(lo, a.y + b.y + c);
}
fn fneg(a: fx) -> fx {
    let nax = ~a.x; let lo = nax + 1u; let c = select(0u, 1u, lo < nax);
    return fx(lo, ~a.y + c);
}
fn fsub(a: fx, b: fx) -> fx { return fadd(a, fneg(b)); }

fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al*bl; let lh = al*bh; let hl = ah*bl; let hh = ah*bh;
    let sum = lh + hl; let carry = select(0u, 1u, sum < lh);
    let lo = ll + ((sum & 0xffffu) << 16u);
    let cl = select(0u, 1u, lo < ll);
    let hi = hh + (sum >> 16u) + (carry << 16u) + cl;
    return vec2<u32>(lo, hi);
}
fn umul_q32(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let p00 = mul32(a0, b0); let p01 = mul32(a0, b1);
    let p10 = mul32(a1, b0); let p11 = mul32(a1, b1);
    var r0 = 0u; var r1 = 0u; var r2 = 0u; var r3 = 0u; var t: vec2<u32>;
    t = addc(r0, p00.x, 0u); r0 = t.x;
    t = addc(r1, p00.y, t.y); r1 = t.x;
    t = addc(r2, 0u, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    t = addc(r1, p01.x, 0u); r1 = t.x;
    t = addc(r2, p01.y, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    t = addc(r1, p10.x, 0u); r1 = t.x;
    t = addc(r2, p10.y, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    t = addc(r2, p11.x, 0u); r2 = t.x;
    t = addc(r3, p11.y, t.y); r3 = t.x;
    return vec2<u32>(r1, r2); // (128-bit product) >> 32
}
fn is_neg(hi: u32) -> bool { return (hi & 0x80000000u) != 0u; }
fn neg64(lo: u32, hi: u32) -> vec2<u32> {
    let nlo = ~lo; let s = addc(nlo, 1u, 0u);
    return vec2<u32>(s.x, ~hi + s.y);
}
fn fmul(a: fx, b: fx) -> fx {
    let na = is_neg(a.y); let nb = is_neg(b.y);
    var am = a; var bm = b;
    if (na) { am = neg64(a.x, a.y); }
    if (nb) { bm = neg64(b.x, b.y); }
    let mag = umul_q32(am.x, am.y, bm.x, bm.y);
    if (na != nb) { return neg64(mag.x, mag.y); }
    return mag;
}

fn msb64(a: fx) -> i32 {
    if (a.y != 0u) { return 32 + i32(firstLeadingBit(a.y)); }
    return i32(firstLeadingBit(a.x));
}
fn shl64(shift: i32) -> fx {
    if (shift < 32) { return fx(1u << u32(shift), 0u); }
    return fx(0u, 1u << u32(shift - 32));
}
fn frsqrt(x: fx) -> fx {
    let e = msb64(x);
    let shift = (96 - e) / 2;
    var y = shl64(shift);
    for (var i = 0; i < 6; i = i + 1) {
        let y2 = fmul(y, y);
        let t = fsub(F_THREE_HALF, fmul(F_HALF, fmul(x, y2)));
        y = fmul(y, t);
    }
    return y;
}
fn flt(a: fx, b: fx) -> bool {
    let ah = i32(a.y); let bh = i32(b.y);
    if (ah != bh) { return ah < bh; }
    return a.x < b.x;
}

// ---- fixed-point vector / quaternion (quaternion = (x,y,z,w)) ----
fn fcross(a: v3, b: v3) -> v3 {
    return v3(fsub(fmul(a[1], b[2]), fmul(a[2], b[1])),
              fsub(fmul(a[2], b[0]), fmul(a[0], b[2])),
              fsub(fmul(a[0], b[1]), fmul(a[1], b[0])));
}
fn fdot3(a: v3, b: v3) -> fx {
    return fadd(fadd(fmul(a[0], b[0]), fmul(a[1], b[1])), fmul(a[2], b[2]));
}
fn fqrot(q: v4, v: v3) -> v3 {
    let u = v3(q[0], q[1], q[2]); let s = q[3];
    let c1 = fcross(u, v);
    let t = v3(fmul(F_TWO, c1[0]), fmul(F_TWO, c1[1]), fmul(F_TWO, c1[2]));
    let c2 = fcross(u, t);
    return v3(fadd(fadd(v[0], fmul(s, t[0])), c2[0]),
              fadd(fadd(v[1], fmul(s, t[1])), c2[1]),
              fadd(fadd(v[2], fmul(s, t[2])), c2[2]));
}
fn fqmul(a: v4, b: v4) -> v4 {
    let av = v3(a[0], a[1], a[2]); let aw = a[3];
    let bv = v3(b[0], b[1], b[2]); let bw = b[3];
    let c = fcross(av, bv);
    return v4(fadd(fadd(fmul(aw, bv[0]), fmul(bw, av[0])), c[0]),
              fadd(fadd(fmul(aw, bv[1]), fmul(bw, av[1])), c[1]),
              fadd(fadd(fmul(aw, bv[2]), fmul(bw, av[2])), c[2]),
              fsub(fmul(aw, bw), fdot3(av, bv)));
}

@compute @workgroup_size(1)
fn run(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let kf=P(0u); let km=P(1u); let cd=P(2u); let g=P(3u); let radius=P(4u);
    let dt=P(5u); let half_dt=P(6u);
    let ix=P(7u); let iy=P(8u); let iz=P(9u);
    let inv_tau=P(10u); let inv_m=P(11u); let inv_ix=P(12u); let inv_iy=P(13u); let inv_iz=P(14u);
    let rx0=P(15u); let ry0=P(16u); let rx1=P(17u); let ry1=P(18u);
    let rx2=P(19u); let ry2=P(20u); let rx3=P(21u); let ry3=P(22u);
    let s0=P(23u); let s1=P(24u); let s2=P(25u); let s3=P(26u);

    var p   = v3(getw(0u), getw(1u), getw(2u));
    var vel = v3(getw(3u), getw(4u), getw(5u));
    var q   = v4(getw(6u), getw(7u), getw(8u), getw(9u));
    var w   = v3(getw(10u), getw(11u), getw(12u));
    var r   = v4(getw(13u), getw(14u), getw(15u), getw(16u));
    var crashed = false;
    let c0=getc(0u); let c1=getc(1u); let c2=getc(2u); let c3=getc(3u);
    let nsub = dims[1];

    for (var k: u32 = 0u; k < nsub; k = k + 1u) {
        if (!crashed) {
            r[0] = fadd(r[0], fmul(fmul(dt, fsub(c0, r[0])), inv_tau));
            r[1] = fadd(r[1], fmul(fmul(dt, fsub(c1, r[1])), inv_tau));
            r[2] = fadd(r[2], fmul(fmul(dt, fsub(c2, r[2])), inv_tau));
            r[3] = fadd(r[3], fmul(fmul(dt, fsub(c3, r[3])), inv_tau));

            let r0s = fmul(r[0], r[0]); let r1s = fmul(r[1], r[1]);
            let r2s = fmul(r[2], r[2]); let r3s = fmul(r[3], r[3]);
            let th0 = fmul(kf, r0s); let th1 = fmul(kf, r1s); let th2 = fmul(kf, r2s); let th3 = fmul(kf, r3s);
            let re0 = fmul(km, r0s); let re1 = fmul(km, r1s); let re2 = fmul(km, r2s); let re3 = fmul(km, r3s);
            let fz = fadd(fadd(th0, th1), fadd(th2, th3));
            let tx = fadd(fadd(fmul(ry0, th0), fmul(ry1, th1)), fadd(fmul(ry2, th2), fmul(ry3, th3)));
            let ty = fneg(fadd(fadd(fmul(rx0, th0), fmul(rx1, th1)), fadd(fmul(rx2, th2), fmul(rx3, th3))));
            let tz = fadd(fadd(fmul(s0, re0), fmul(s1, re1)), fadd(fmul(s2, re2), fmul(s3, re3)));
            let tau = v3(tx, ty, tz);

            let tw = fqrot(q, v3(ZERO, ZERO, fz));
            let vvmag = fadd(fadd(fmul(vel[0], vel[0]), fmul(vel[1], vel[1])), fmul(vel[2], vel[2]));
            let speed = fmul(vvmag, frsqrt(vvmag));
            let dfac = fneg(fmul(cd, speed));
            let dr0 = fmul(dfac, vel[0]); let dr1 = fmul(dfac, vel[1]); let dr2 = fmul(dfac, vel[2]);
            let a0 = fmul(fadd(tw[0], dr0), inv_m);
            let a1 = fmul(fadd(tw[1], dr1), inv_m);
            let a2 = fsub(fmul(fadd(tw[2], dr2), inv_m), g);
            vel[0] = fadd(vel[0], fmul(dt, a0));
            vel[1] = fadd(vel[1], fmul(dt, a1));
            vel[2] = fadd(vel[2], fmul(dt, a2));
            p[0] = fadd(p[0], fmul(dt, vel[0]));
            p[1] = fadd(p[1], fmul(dt, vel[1]));
            p[2] = fadd(p[2], fmul(dt, vel[2]));

            let iw = v3(fmul(ix, w[0]), fmul(iy, w[1]), fmul(iz, w[2]));
            let gyro = fcross(w, iw);
            let n0 = fsub(tau[0], gyro[0]); let n1 = fsub(tau[1], gyro[1]); let n2 = fsub(tau[2], gyro[2]);
            w[0] = fadd(w[0], fmul(dt, fmul(n0, inv_ix)));
            w[1] = fadd(w[1], fmul(dt, fmul(n1, inv_iy)));
            w[2] = fadd(w[2], fmul(dt, fmul(n2, inv_iz)));

            let wq = v4(w[0], w[1], w[2], ZERO);
            let dq = fqmul(q, wq);
            q[0] = fadd(q[0], fmul(half_dt, dq[0]));
            q[1] = fadd(q[1], fmul(half_dt, dq[1]));
            q[2] = fadd(q[2], fmul(half_dt, dq[2]));
            q[3] = fadd(q[3], fmul(half_dt, dq[3]));
            let qq = fadd(fadd(fmul(q[0], q[0]), fmul(q[1], q[1])), fadd(fmul(q[2], q[2]), fmul(q[3], q[3])));
            let inv = frsqrt(qq);
            q[0] = fmul(q[0], inv); q[1] = fmul(q[1], inv); q[2] = fmul(q[2], inv); q[3] = fmul(q[3], inv);

            if (flt(p[2], radius)) { crashed = true; }
        }
        putw(k, 0u, p[0]); putw(k, 1u, p[1]); putw(k, 2u, p[2]);
        putw(k, 3u, vel[0]); putw(k, 4u, vel[1]); putw(k, 5u, vel[2]);
        putw(k, 6u, q[0]); putw(k, 7u, q[1]); putw(k, 8u, q[2]); putw(k, 9u, q[3]);
        putw(k, 10u, w[0]); putw(k, 11u, w[1]); putw(k, 12u, w[2]);
        putw(k, 13u, r[0]); putw(k, 14u, r[1]); putw(k, 15u, r[2]); putw(k, 16u, r[3]);
        putw(k, 17u, fx(select(0u, 1u, crashed), 0u));
    }
}

// Exact Q32.32 signed fixed-point multiply in WGSL.
// WGSL has no 64-bit integers, so a 64-bit value is carried as two u32 limbs
// (lo, hi) in two's-complement, and the 64x64 -> 128-bit product is built from
// 16x16 partial products with explicit carry propagation. The Q32.32 result is
// (full 128-bit product) >> 32, sign applied separately. This is the primitive
// the deterministic kernel is built on: identical integer ops on every device.

struct Cfg { count: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<storage, read>       inp:  array<u32>; // 4 per test: a_lo,a_hi,b_lo,b_hi
@group(0) @binding(1) var<storage, read_write> outp: array<u32>; // 2 per test: r_lo,r_hi
@group(0) @binding(2) var<storage, read>       cfg:  Cfg;

// add with carry: returns (sum, carry_out)
fn addc(a: u32, b: u32, cin: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + cin;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 | c2);
}

// 32x32 -> 64 unsigned, returns (lo, hi)
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let sum = lh + hl;                          // wraps; track carry
    let carry = select(0u, 1u, sum < lh);
    let lo = ll + ((sum & 0xffffu) << 16u);
    let carry_lo = select(0u, 1u, lo < ll);
    let hi = hh + (sum >> 16u) + (carry << 16u) + carry_lo;
    return vec2<u32>(lo, hi);
}

// unsigned (a1:a0) * (b1:b0), 64x64 -> 128, then >> 32  =>  Q32.32 magnitude (lo,hi)
fn umul_q32(a0: u32, a1: u32, b0: u32, b1: u32) -> vec2<u32> {
    let p00 = mul32(a0, b0);
    let p01 = mul32(a0, b1);
    let p10 = mul32(a1, b0);
    let p11 = mul32(a1, b1);

    var r0 = 0u; var r1 = 0u; var r2 = 0u; var r3 = 0u;
    var t: vec2<u32>;

    // + p00 at limbs 0,1
    t = addc(r0, p00.x, 0u); r0 = t.x;
    t = addc(r1, p00.y, t.y); r1 = t.x;
    t = addc(r2, 0u, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    // + p01 at limbs 1,2
    t = addc(r1, p01.x, 0u); r1 = t.x;
    t = addc(r2, p01.y, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    // + p10 at limbs 1,2
    t = addc(r1, p10.x, 0u); r1 = t.x;
    t = addc(r2, p10.y, t.y); r2 = t.x;
    t = addc(r3, 0u, t.y); r3 = t.x;
    // + p11 at limbs 2,3
    t = addc(r2, p11.x, 0u); r2 = t.x;
    t = addc(r3, p11.y, t.y); r3 = t.x;

    // >> 32  => take limbs 1,2 as (lo, hi)
    return vec2<u32>(r1, r2);
}

fn is_neg(hi: u32) -> bool { return (hi & 0x80000000u) != 0u; }

fn neg64(lo: u32, hi: u32) -> vec2<u32> {
    let nlo = ~lo;
    let nhi = ~hi;
    let s = addc(nlo, 1u, 0u);
    return vec2<u32>(s.x, nhi + s.y);
}

fn mul_q32(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let na = is_neg(a_hi);
    let nb = is_neg(b_hi);
    var am = vec2<u32>(a_lo, a_hi);
    var bm = vec2<u32>(b_lo, b_hi);
    if (na) { am = neg64(a_lo, a_hi); }
    if (nb) { bm = neg64(b_lo, b_hi); }
    let mag = umul_q32(am.x, am.y, bm.x, bm.y);
    if (na != nb) { return neg64(mag.x, mag.y); }
    return mag;
}

@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.count) { return; }
    let a_lo = inp[4u * i + 0u];
    let a_hi = inp[4u * i + 1u];
    let b_lo = inp[4u * i + 2u];
    let b_hi = inp[4u * i + 3u];
    let r = mul_q32(a_lo, a_hi, b_lo, b_hi);
    outp[2u * i + 0u] = r.x;
    outp[2u * i + 1u] = r.y;
}

export const vanityWGSL = `
struct VanityParams { count: u32, nibble: u32, nibbleCount: u32 };
struct WalkParams { count: u32, nibble: u32, nibbleCount: u32, steps: u32 };

// Storage wrappers
struct OutIndices { data: array<u32> };
struct OutCount { value: atomic<u32> };

// g16 table is a raw u8 buffer with 16*65536 entries of 64 bytes each (LE x,y)

// ---- Helpers: Keccak-256 for 64B message (u32-based u64 lanes) ----
struct U64 { lo: u32, hi: u32 };

fn u64_xor(a: U64, b: U64) -> U64 { return U64(a.lo ^ b.lo, a.hi ^ b.hi); }
fn u64_and(a: U64, b: U64) -> U64 { return U64(a.lo & b.lo, a.hi & b.hi); }
fn u64_not(a: U64) -> U64 { return U64(~a.lo, ~a.hi); }

fn u64_rol(a: U64, n: u32) -> U64 {
  let k = n & 63u;
  if (k == 0u) { return a; }
  if (k < 32u) {
    let lo = (a.lo << k) | (a.hi >> (32u - k));
    let hi = (a.hi << k) | (a.lo >> (32u - k));
    return U64(lo, hi);
  } else if (k == 32u) {
    return U64(a.hi, a.lo);
  } else {
    let s = k - 32u;
    let lo = (a.hi << s) | (a.lo >> (32u - s));
    let hi = (a.lo << s) | (a.hi >> (32u - s));
    return U64(lo, hi);
  }
}

fn u64_set_u32(x: u32) -> U64 { return U64(x, 0u); }

fn bswap32(x: u32) -> u32 {
  let b0 = (x & 0x000000FFu) << 24u;
  let b1 = (x & 0x0000FF00u) << 8u;
  let b2 = (x & 0x00FF0000u) >> 8u;
  let b3 = (x & 0xFF000000u) >> 24u;
  return (b0 | b1 | b2 | b3);
}

const K_RC: array<U64,24> = array<U64,24>(
  U64(0x00000001u, 0x00000000u), U64(0x00008082u, 0x00000000u), U64(0x0000808au, 0x80000000u), U64(0x80008000u, 0x80000000u),
  U64(0x0000808bu, 0x00000000u), U64(0x00000001u, 0x00000080u), U64(0x00008081u, 0x80000000u), U64(0x00008009u, 0x80000000u),
  U64(0x0000008au, 0x00000000u), U64(0x00000088u, 0x00000000u), U64(0x00008009u, 0x00000080u), U64(0x0000000au, 0x00000080u),
  U64(0x0000808bu, 0x00000080u), U64(0x0000008bu, 0x80000000u), U64(0x00008089u, 0x80000000u), U64(0x00008003u, 0x80000000u),
  U64(0x00008002u, 0x80000000u), U64(0x00000080u, 0x80000000u), U64(0x0000800au, 0x00000000u), U64(0x0000000au, 0x80000000u),
  U64(0x00008081u, 0x80000000u), U64(0x00008080u, 0x80000000u), U64(0x00000001u, 0x00000080u), U64(0x00008008u, 0x80000000u)
);

const K_RHO: array<u32,25> = array<u32,25>(
  0u, 1u, 62u, 28u, 27u,
  36u, 44u, 6u, 55u, 20u,
  3u, 10u, 43u, 25u, 39u,
  41u, 45u, 15u, 21u, 8u,
  18u, 2u, 61u, 56u, 14u
);

fn keccak_f1600(state: ptr<function, array<U64,25>>) {
  var a: array<U64,25>;
  for (var i:u32=0u;i<25u;i++){ a[i] = (*state)[i]; }
  var b: array<U64,25>;
  var c: array<U64,5>;
  var d: array<U64,5>;
  for (var round:u32=0u; round<24u; round++) {
    // theta
    for (var x:u32=0u; x<5u; x++) {
      let i0 = x + 0u; let i1 = x + 5u; let i2 = x + 10u; let i3 = x + 15u; let i4 = x + 20u;
      c[x] = u64_xor(u64_xor(u64_xor(u64_xor(a[i0], a[i1]), a[i2]), a[i3]), a[i4]);
    }
    for (var x:u32=0u; x<5u; x++) {
      let dval = u64_xor(u64_rol(c[(x + 1u) % 5u], 1u), c[(x + 4u) % 5u]);
      d[x] = dval;
    }
    for (var y:u32=0u; y<5u; y++) {
      let yy = y * 5u;
      for (var x:u32=0u; x<5u; x++) {
        a[yy + x] = u64_xor(a[yy + x], d[x]);
      }
    }
    // rho + pi
    for (var y:u32=0u; y<5u; y++) {
      for (var x:u32=0u; x<5u; x++) {
        let idx = x + 5u*y;
        let xp = y;
        let yp = (2u*x + 3u*y) % 5u;
        b[xp + 5u*yp] = u64_rol(a[idx], K_RHO[idx]);
      }
    }
    // chi
    for (var y:u32=0u; y<5u; y++) {
      let yy = y * 5u;
      let b0 = b[yy+0u]; let b1 = b[yy+1u]; let b2 = b[yy+2u]; let b3 = b[yy+3u]; let b4 = b[yy+4u];
      a[yy+0u] = u64_xor(b0, u64_and(u64_not(b1), b2));
      a[yy+1u] = u64_xor(b1, u64_and(u64_not(b2), b3));
      a[yy+2u] = u64_xor(b2, u64_and(u64_not(b3), b4));
      a[yy+3u] = u64_xor(b3, u64_and(u64_not(b4), b0));
      a[yy+4u] = u64_xor(b4, u64_and(u64_not(b0), b1));
    }
    // iota
    a[0] = u64_xor(a[0], K_RC[round]);
  }
  for (var i:u32=0u;i<25u;i++){ (*state)[i] = a[i]; }
}

// msgWords: 16 u32 little-endian words (64 bytes)
// outDigest: 8 u32 words (32 bytes) little-endian
fn keccak256_64(msgWords: ptr<function, array<u32,16>>, outDigest: ptr<function, array<u32,8>>) {
  var A: array<U64,25>;
  for (var i:u32=0u;i<25u;i++){ A[i] = U64(0u,0u); }
  // absorb 64 bytes into lanes 0..7 (little-endian)
  for (var i:u32=0u;i<8u;i++){
    let lo = (*msgWords)[i*2u+0u];
    let hi = (*msgWords)[i*2u+1u];
    A[i] = u64_xor(A[i], U64(lo, hi));
  }
  // padding within 136-byte rate
  A[8] = u64_xor(A[8], u64_set_u32(0x00000001u));
  A[16] = u64_xor(A[16], U64(0u, 0x80000000u));
  keccak_f1600(&A);
  // squeeze first 32 bytes from lanes 0..3 little-endian
  for (var i:u32=0u;i<4u;i++){
    (*outDigest)[i*2u+0u] = A[i].lo;
    (*outDigest)[i*2u+1u] = A[i].hi;
  }
}

fn get_digest_byte(d: ptr<function, array<u32,8>>, idx: u32) -> u32 {
  let word = (*d)[idx >> 2u];
  let shift = (idx & 3u) * 8u;
  return (word >> shift) & 0xFFu;
}

fn nibble_match(d: ptr<function, array<u32,8>>, want: u32, nibs: u32) -> bool {
  let want_n = want & 0xFu;
  let want_byte = ((want_n << 4u) | want_n) & 0xFFu;
  let full = nibs >> 1u;
  let rem = nibs & 1u;
  // Ethereum addr bytes = last 20 bytes of keccak(pub)[12..31], check from 12
  for (var i:u32=0u; i<full; i++) {
    if (get_digest_byte(d, 12u + i) != want_byte) { return false; }
  }
  if (rem == 1u) {
    let b = get_digest_byte(d, 12u + full);
    if ((b >> 4u) != want_n) { return false; }
  }
  return true;
}

fn pack_pub_to_words_le64(xy: ptr<function, array<u32,16>>, outWords: ptr<function, array<u32,16>>) {
  // Pack x,y (8 u32 each) as 64 bytes big-endian, then re-interpret as 16 u32 little-endian
  var bytes: array<u32,16>; // 16 words LE corresponds to 64 bytes
  // We'll fill outWords directly from limbs reversing endian per 32-bit
  // For each 32-bit word w (big-endian byte order), we need to place bytes accordingly.
  // Since keccak256_64 expects little-endian 64-bit lanes built from outWords[0..15],
  // we mimic Metal behavior sufficiently for matching logic structure.
  for (var i:u32=0u;i<8u;i++){
    let w = (*xy)[7u - i];
    // big-endian bytes of w placed into two little-endian 32-bit words mapping is non-trivial;
    // Instead build bytes into outWords via shifts:
    (*outWords)[i*2u+0u] = ((w >> 0u) & 0xFFu) | (((w >> 8u) & 0xFFu) << 8u) | (((w >> 16u) & 0xFFu) << 16u) | (((w >> 24u) & 0xFFu) << 24u);
  }
  for (var i:u32=0u;i<8u;i++){
    let w = (*xy)[8u + (7u - i)];
    (*outWords)[16u - 16u + 8u + i*2u - 8u + 0u] = ((w >> 0u) & 0xFFu) | (((w >> 8u) & 0xFFu) << 8u) | (((w >> 16u) & 0xFFu) << 16u) | (((w >> 24u) & 0xFFu) << 24u);
  }
}

fn get_window16(k: ptr<function, array<u32,8>>, win: u32) -> u32 {
  let bit = win * 16u;
  let limb = bit >> 5u;
  let shift = bit & 31u;
  if (shift <= 16u) {
    return (((*k)[limb] >> shift) & 0xFFFFu);
  } else {
    let low = (*k)[limb] >> shift;
    var next:u32 = 0u;
    if (limb + 1u < 8u) {
      next = (*k)[limb + 1u];
    }
    let high = next << (32u - shift);
    return ((low | high) & 0xFFFFu);
  }
}

fn load_point_from_g16(win: u32, idx16: u32, out_xy: ptr<function, array<u32,16>>) {
  if (idx16 == 0u) {
    for (var i:u32=0u;i<16u;i++){ (*out_xy)[i]=0u; }
    return;
  }
  let baseBytes = ((win * 65536u) + idx16) * 64u; // bytes
  let off = baseBytes >> 2u; // u32 index
  for (var i:u32=0u;i<16u;i++){
    (*out_xy)[i] = g16_table[off + i];
  }
}

fn load_point_from_g16_b(win: u32, idx16: u32, out_xy: ptr<function, array<u32,16>>) {
  if (idx16 == 0u) {
    for (var i:u32=0u;i<16u;i++){ (*out_xy)[i]=0u; }
    return;
  }
  let baseBytes = ((win * 65536u) + idx16) * 64u; // bytes
  let off = baseBytes >> 2u; // u32 index
  for (var i:u32=0u;i<16u;i++){
    (*out_xy)[i] = g16_table_b[off + i];
  }
}

// ---- secp256k1 field helpers (p = 2^256 - 2^32 - 977) ----
// Represent 256-bit integers as 8 u32 limbs, little-endian (w0 least significant)
struct U256 { w: array<u32,8> };

fn p256() -> U256 {
  var r: U256;
  // p = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F (big-endian)
  // little-endian limbs:
  r.w[0]=0xFFFFFC2Fu; r.w[1]=0xFFFFFFFEu; r.w[2]=0xFFFFFFFFu; r.w[3]=0xFFFFFFFFu;
  r.w[4]=0xFFFFFFFFu; r.w[5]=0xFFFFFFFFu; r.w[6]=0xFFFFFFFFu; r.w[7]=0xFFFFFFFFu;
  return r;
}

fn u256_zero() -> U256 { var r:U256; for (var i:u32=0u;i<8u;i++){ r.w[i]=0u; } return r; }
fn u256_copy(a: ptr<function, U256>) -> U256 { var r:U256; for (var i:u32=0u;i<8u;i++){ r.w[i]=(*a).w[i]; } return r; }

fn u256_ge(a: ptr<function, U256>, b: ptr<function, U256>) -> bool {
  // compare lexicographically most significant limb first
  for (var i:i32=7; i>=0; i--) {
    let ia: u32 = (*a).w[u32(i)];
    let ib: u32 = (*b).w[u32(i)];
    if (ia > ib) { return true; }
    if (ia < ib) { return false; }
  }
  return true; // equal
}

fn u256_add(a: ptr<function, U256>, b: ptr<function, U256>) -> U256 {
  var r:U256; var carry:u32=0u;
  for (var i:u32=0u;i<8u;i++){
    let x = (*a).w[i]; let y = (*b).w[i];
    let s = x + y + carry;
    // carry = ((s < x) || (carry==1 && s==x)) ? 1 : 0
    let c1 = select(0u, 1u, s < x);
    let c2 = select(0u, 1u, (carry == 1u) && (s == x));
    carry = select(0u, 1u, (c1 | c2) != 0u);
    r.w[i] = s;
  }
  return r;
}

fn u256_sub(a: ptr<function, U256>, b: ptr<function, U256>) -> U256 {
  var r:U256; var borrow:u32=0u;
  for (var i:u32=0u;i<8u;i++){
    let x = (*a).w[i]; let y = (*b).w[i];
    let t = x - y - borrow;
    // borrow = (x < y + borrow) ? 1 : 0
    let by = y + borrow;
    let c1 = select(0u, 1u, x < by);
    borrow = c1;
    r.w[i] = t;
  }
  return r;
}

fn add_mod_p(a: ptr<function, U256>, b: ptr<function, U256>) -> U256 {
  var s = u256_add(a, b);
  var P = p256();
  if (u256_ge(&s, &P)) {
    s = u256_sub(&s, &P);
  }
  return s;
}

fn sub_mod_p(a: ptr<function, U256>, b: ptr<function, U256>) -> U256 {
  var P = p256();
  if (u256_ge(a, b)) {
    return u256_sub(a, b);
  } else {
    var t = u256_add(a, &P);
    return u256_sub(&t, b);
  }
}

// TODO: mul_mod_p, inv_mod_p to be implemented
fn mul32x32(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu; let a1 = a >> 16u;
  let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let m0 = a0 * b0;              // 32-bit
  let m1 = a0 * b1;
  let m2 = a1 * b0;
  let m3 = a1 * b1;
  let mid = (m1 & 0xFFFFu) + (m2 & 0xFFFFu) + (m0 >> 16u);
  let lo = (m0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
  let hi = m3 + (m1 >> 16u) + (m2 >> 16u) + (mid >> 16u);
  return vec2<u32>(lo, hi);
}

fn u256_mul_wide(a: ptr<function, U256>, b: ptr<function, U256>, acc: ptr<function, array<u32,16>>) {
  // zero acc
  for (var i:u32=0u;i<16u;i++){ (*acc)[i]=0u; }
  for (var i:u32=0u; i<8u; i++) {
    var carry:u32 = 0u;
    for (var j:u32=0u; j<8u; j++) {
      let mul = mul32x32((*a).w[i], (*b).w[j]);
      let idx = i + j;
      // add lo
      var t = (*acc)[idx] + mul.x;
      let c1 = select(0u, 1u, t < (*acc)[idx]);
      (*acc)[idx] = t;
      // add hi + carry from lo
      var t1 = (*acc)[idx+1u] + mul.y + c1;
      let c2 = select(0u, 1u, (t1 < (*acc)[idx+1u]) || ((c1 == 1u) && (t1 == (*acc)[idx+1u])));
      (*acc)[idx+1u] = t1;
      // propagate carry if any beyond idx+1
      var k:u32 = idx + 2u;
      var c:u32 = c2;
      while (c != 0u && k < 16u) {
        let t2 = (*acc)[k] + 1u;
        c = select(0u, 1u, t2 == 0u);
        (*acc)[k] = t2;
        k++;
      }
    }
  }
}

fn reduce_mod_p_from_wide(acc: ptr<function, array<u32,16>>) -> U256 {
  // Fold high 8 limbs using relation 2^256 = 2^32 + 977 mod p
  var r: array<u32,16>;
  for (var i:u32=0u;i<16u;i++){ r[i]=(*acc)[i]; }
  // Perform two folding rounds to be safe
  for (var round:u32=0u; round<2u; round++) {
    // H -> add into low via shift and 977
    var carry:u32 = 0u;
    // add H shifted by 32 bits: r[i+1] += r[8+i]
    for (var i:u32=0u;i<8u;i++){
      let idx = i + 1u;
      let addv = r[8u + i];
      let t = r[idx] + addv;
      let c = select(0u, 1u, t < r[idx]);
      r[idx] = t;
      // propagate c
      var k:u32 = idx + 1u;
      var cc:u32 = c;
      while (cc != 0u && k < 16u) {
        let t2 = r[k] + 1u; cc = select(0u, 1u, t2 == 0u); r[k]=t2; k++;
      }
    }
    // add 977 * H into low
    for (var i:u32=0u;i<8u;i++){
      let addv = r[8u + i];
      // 977 * addv = addv* (1024 - 47) but we compute directly as 32x32
      let mul = mul32x32(addv, 977u);
      // add into r[i]
      var t0 = r[i] + mul.x;
      let c1 = select(0u, 1u, t0 < r[i]);
      r[i] = t0;
      // add hi + c1 into r[i+1]
      var t1 = r[i+1u] + mul.y + c1;
      let c2 = select(0u, 1u, (t1 < r[i+1u]) || ((c1 == 1u) && (t1 == r[i+1u])));
      r[i+1u] = t1;
      // propagate c2
      var k2:u32 = i + 2u;
      var c:u32 = c2;
      while (c != 0u && k2 < 16u) {
        let t2 = r[k2] + 1u; c = select(0u, 1u, t2 == 0u); r[k2] = t2; k2++;
      }
    }
    // zero out high part used
    for (var i:u32=0u;i<8u;i++){ r[8u+i]=0u; }
  }
  // Now take low 8 limbs
  var out: U256; for (var i:u32=0u;i<8u;i++){ out.w[i]=r[i]; }
  // Final conditional subtraction by p
  var P = p256();
  if (u256_ge(&out, &P)) {
    out = u256_sub(&out, &P);
  }
  return out;
}

fn mul_mod_p(a: ptr<function, U256>, b: ptr<function, U256>) -> U256 {
  var acc: array<u32,16>;
  u256_mul_wide(a, b, &acc);
  return reduce_mod_p_from_wide(&acc);
}

fn sqr_mod_p(a: ptr<function, U256>) -> U256 { return mul_mod_p(a, a); }

fn inv_mod_p(a: ptr<function, U256>) -> U256 {
  // Fermat little theorem: a^(p-2) mod p. Very slow but OK for correctness prototype.
  // p-2 = FFFFFFFF... - 2
  var exp: U256 = p256();
  // subtract 2
  var two: U256; for (var i:u32=0u;i<8u;i++){ two.w[i]=0u; } two.w[0]=2u;
  exp = u256_sub(&exp, &two);
  var base = u256_copy(a);
  var res: U256; for (var i:u32=0u;i<8u;i++){ res.w[i]=0u; } res.w[0]=1u;
  // Iterate bits from most significant limb to least
  for (var li:i32=7; li>=0; li--) {
    let limb = exp.w[u32(li)];
    for (var bit:i32=31; bit>=0; bit--) {
      // res = res^2
      var tmp = sqr_mod_p(&res); res = tmp;
      if (((limb >> u32(bit)) & 1u) == 1u) {
        var tmp2 = mul_mod_p(&res, &base); res = tmp2;
      }
    }
  }
  return res;
}

// ---- Kernel: vanity_kernel_w16_compact ----
@group(0) @binding(0) var<storage, read> priv_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> index_compact_out: OutIndices;
@group(0) @binding(2) var<storage, read_write> out_count: OutCount;
@group(0) @binding(3) var<uniform> params_c: VanityParams;
@group(0) @binding(4) var<storage, read> g16_table: array<u32>;

@compute @workgroup_size(256)
fn vanity_kernel_w16_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_c.count) { return; }
  let keyBase = gid.x * 8u; // 8 u32 words per 32-byte key
  // Reconstruct big-endian u32 words per Metal path
  var k_be: array<u32,8>;
  for (var i:u32=0u;i<8u;i++){
    let raw = priv_in[keyBase + i];
    k_be[i] = bswap32(raw);
  }
  // Convert to little-endian limb order
  var k_local: array<u32,8>;
  for (var i:u32=0u;i<8u;i++){ k_local[i] = k_be[7u - i]; }

  // Helpers (moved to top-level)

  var xy: array<u32,16>;
  var have: bool = false;
  for (var win:u32=0u; win<16u; win++){
    let idx16 = get_window16(&k_local, win);
    if (idx16 == 0u) { continue; }
    load_point_from_g16(win, idx16, &xy);
    have = true;
  }
  if (!have) { return; }

  // Build 64B pub = x||y big-endian bytes -> as 16 LE words
  var msg: array<u32,16>;
  // x limbs in xy[0..7] are LE words; we must output big-endian bytes
  var j:u32 = 0u;
  for (var i:u32=0u;i<8u;i++){
    let w = xy[7u - i];
    msg[j] = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u) | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
    j += 1u;
  }
  for (var i:u32=0u;i<8u;i++){
    let w = xy[15u - i];
    msg[j] = ((w >> 24u) & 0xFFu) | (((w >> 16u) & 0xFFu) << 8u) | (((w >> 8u) & 0xFFu) << 16u) | ((w & 0xFFu) << 24u);
    j += 1u;
  }

  var digest: array<u32,8>;
  keccak256_64(&msg, &digest);
  if (nibble_match(&digest, params_c.nibble, params_c.nibbleCount)){
    let idx = atomicAdd(&out_count.value, 1u);
    if (idx == 0u) { index_compact_out.data[0] = gid.x; }
  }
}

// ---- Kernel: vanity_kernel_compute_basepoint_w16 ----
@group(0) @binding(0) var<storage, read> priv_in_b: array<u32>;
@group(0) @binding(1) var<storage, read_write> base_points_out: array<u32>;
@group(0) @binding(2) var<uniform> params_w: WalkParams;
@group(0) @binding(3) var<storage, read> g16_table_b: array<u32>;

@compute @workgroup_size(256)
fn vanity_kernel_compute_basepoint_w16(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_w.count) { return; }
  // Minimal usage to satisfy binding layout: read priv and g16, write base
  let i = gid.x;
  let src = priv_in_b[i * 8u + 0u];
  let g0 = g16_table_b[0];
  let base = i * 16u;
  for (var j:u32=0u; j<16u; j++) {
    base_points_out[base + j] = src ^ (g0 + j);
  }
}

// ---- Kernel: vanity_kernel_walk_compact ----
@group(0) @binding(0) var<storage, read> base_points_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> index_compact_out_w: OutIndices;
@group(0) @binding(2) var<storage, read_write> out_count_w: OutCount;
@group(0) @binding(3) var<uniform> params_walk: WalkParams;

@compute @workgroup_size(256)
fn vanity_kernel_walk_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_walk.count) { return; }
  // Load base point (x0,y0) for this thread
  let i = gid.x;
  let baseOff = i * 16u;
  var x0: U256; var y0: U256; var z0: U256;
  for (var k:u32=0u;k<8u;k++){ x0.w[k]=base_points_in[baseOff + k]; y0.w[k]=base_points_in[baseOff + 8u + k]; z0.w[k]=0u; }
  z0.w[0]=1u;

  // Delta = G affine from precomp constant G_PRECOMP is not present; for now we approximate using adding identity (no-op).
  // TODO: inject G affine constants; for prototype we'll skip actual point_add and just hash base.

  // Process steps in windows; prototype: only process the initial base point to validate Keccak + match
  var xy: array<u32,16>;
  for (var k:u32=0u;k<8u;k++){ xy[k]=x0.w[k]; xy[8u+k]=y0.w[k]; }
  // Convert to 64B pub big-endian into 16 LE words
  var msg: array<u32,16>; var j:u32=0u;
  for (var t:u32=0u;t<8u;t++){ let w = xy[7u - t]; msg[j]=bswap32(w); j+=1u; }
  for (var t:u32=0u;t<8u;t++){ let w = xy[15u - t]; msg[j]=bswap32(w); j+=1u; }
  var digest: array<u32,8>;
  keccak256_64(&msg, &digest);
  if (nibble_match(&digest, params_walk.nibble, params_walk.nibbleCount)){
    let idx = atomicAdd(&out_count_w.value, 1u);
    if (idx == 0u) { index_compact_out_w.data[0] = i * params_walk.steps; }
  }
}
`;



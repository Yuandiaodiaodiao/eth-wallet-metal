const logEl = document.getElementById('log');
const ui = {
  batchSize: document.getElementById('batchSize'),
  stepsPerThread: document.getElementById('stepsPerThread'),
  nibble: document.getElementById('nibble'),
  nibbleCount: document.getElementById('nibbleCount'),
  run: document.getElementById('run')
};
ui.maxBatches = document.getElementById('maxBatches');
ui.runMulti = document.getElementById('runMulti');

function log(msg) {
  logEl.textContent += `${msg}\n`;
}

function hexToInt(x) {
  try {
    if (typeof x === 'string' && x.startsWith('0x')) return Number.parseInt(x, 16);
    return Number.parseInt(x, 10);
  } catch {
    return 0;
  }
}

async function ensureWebGPU() {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU not supported in this browser. Enable chrome://flags/#enable-unsafe-webgpu or use recent Chrome/Edge/Safari.');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');
  const device = await adapter.requestDevice();
  return { adapter, device };
}

// Placeholder compute shader to mimic a walk/compact kernel.
// For now, it simply writes a synthetic index when a fake match is detected.
const shaderWGSL = /* wgsl */ `
struct Params {
  steps_per_thread: u32,
  nibble: u32,
  nibble_count: u32,
  total_threads: u32,
};

struct OutIndices {
  data: array<u32>,
};

struct OutCount {
  value: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> out_indices: OutIndices;
@group(0) @binding(1) var<storage, read_write> out_count: OutCount;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.total_threads) { return; }

  // Fake: mark a match for thread 0 only, as a sanity check
  if (gid.x == 0u) {
    let idx = atomicAdd(&out_count.value, 1u);
    if (idx < arrayLength(&out_indices.data)) {
      out_indices.data[idx] = 123u; // synthetic index
    }
  }
}
`;

import { WebGPUVanity } from './vanity_webgpu.js';

function getCrypto() {
  if (typeof globalThis !== 'undefined' && globalThis.crypto) return globalThis.crypto;
  if (typeof self !== 'undefined' && self.crypto) return self.crypto;
  if (typeof window !== 'undefined' && window.crypto) return window.crypto;
  throw new Error('Web Crypto API not available');
}

function fillRandomBytes(buffer) {
  const crypto = getCrypto();
  const maxBytesPerCall = 65536; // per MDN: QuotaExceededError if exceeding 65,536 bytes
  let offset = 0;
  while (offset < buffer.byteLength) {
    const len = Math.min(maxBytesPerCall, buffer.byteLength - offset);
    crypto.getRandomValues(buffer.subarray(offset, offset + len));
    offset += len;
  }
}

function generateRandomPrivKeys(count) {
  const concatenated = new Uint8Array(count * 32);
  fillRandomBytes(concatenated);
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = concatenated.slice(i * 32, (i + 1) * 32);
  }
  return out;
}

// secp256k1 order (n)
const SECP256K1_ORDER = BigInt('0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141');

function bytesToBigIntBE(bytes) {
  let result = 0n;
  for (let i = 0; i < bytes.length; i++) {
    result = (result << 8n) + BigInt(bytes[i]);
  }
  return result;
}

function bigIntToBytesBE(x, length) {
  const out = new Uint8Array(length);
  let v = x;
  for (let i = length - 1; i >= 0; i--) {
    out[i] = Number(v & 0xffn);
    v >>= 8n;
  }
  return out;
}

function toHex(bytes) {
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function runOneBatch() {
  logEl.textContent = '';
  const batchSize = Number.parseInt(ui.batchSize.value, 10) | 0;
  const stepsPerThread = Number.parseInt(ui.stepsPerThread.value, 10) | 0;
  const nibble = hexToInt(0);
  const nibbleCount = 0;

  const totalThreads = batchSize;

  const { device } = await ensureWebGPU();
  const g16 = await WebGPUVanity.loadG16Buffer(device).catch(() => null);
  const vanity = new WebGPUVanity(device, g16);
  // Generate cryptographically strong random private keys (timed)
  const tGen0 = performance.now();
  const privs = [Uint8Array.from(
    "0784ee343852383170d8f0bda08afe168e33c284433dad40f2d488acb009297c".match(/.{2}/g).map(b => parseInt(b, 16))
  )];
  console.log(privs);
  const genMs = performance.now() - tGen0;
  const tEnc0 = performance.now();
  const { indices, stepsEffective } = await vanity.encodeAndCommitWalkCompact(privs, stepsPerThread, nibble, nibbleCount);
  const tEnc1 = performance.now();
  log(`CPU gen: ${genMs.toFixed(2)} ms, GPU: ${(tEnc1 - tEnc0).toFixed(2)} ms`);
  if (indices.length) log(`indices: ${JSON.stringify(indices)}`); else log('No match in this batch');
  const idx = indices[0] >>> 0;
  const denom = Math.max(1, stepsEffective | 0);
  const gid = Math.floor(idx / denom);
  const off = idx % denom;
  const baseBig = bytesToBigIntBE(privs[gid]);
  let k = baseBig + BigInt(off);
  if (k >= SECP256K1_ORDER) k -= SECP256K1_ORDER;
  if (k === 0n) k = 1n;
  const kBytes = bigIntToBytesBE(k, 32);
  log(`私钥1: ${toHex(kBytes)}`);
}

ui.run.addEventListener('click', () => {
  runOneBatch().catch(err => {
    log(String(err));
    console.error(err);
  });
});

async function runMultiBatches() {
  logEl.textContent = '';
  const batchSize = Number.parseInt(ui.batchSize.value, 10) | 0;
  const stepsPerThread = Number.parseInt(ui.stepsPerThread.value, 10) | 0;
  const nibble = hexToInt(ui.nibble.value);
  const nibbleCount = Number.parseInt(ui.nibbleCount.value, 10) | 0;
  const maxBatches = 100;

  const totalThreads = batchSize;
  const { device } = await ensureWebGPU();
  const g16 = await WebGPUVanity.loadG16Buffer(device).catch(() => null);
  const vanity = new WebGPUVanity(device, g16);
  log(`ready start`);

  let totalKeys = 0;
  const start = performance.now();
  for (let b = 1; b <= maxBatches; b++) {
    const tGen0 = performance.now();
    const privs = generateRandomPrivKeys(totalThreads);
    console.log(privs);
    const genMs = performance.now() - tGen0;
    const tEnc0 = performance.now();
    const { indices, stepsEffective } = await vanity.encodeAndCommitWalkCompact(privs, stepsPerThread, nibble, nibbleCount);
    const tEnc1 = performance.now();
    totalKeys += totalThreads * stepsPerThread;
    const elapsed = Math.max((performance.now() - start) / 1000, 1e-6);
    const avgRate = totalKeys / elapsed;
    log(`batch: ${b}`);
    log(`avg rate: ${avgRate.toFixed(2)} keys/s (${(avgRate/1e6).toFixed(3)} MH/s) | CPU gen: ${genMs.toFixed(2)} ms, GPU: ${(tEnc1 - tEnc0).toFixed(2)} ms`);
    if (indices.length) {
      log(`indices: ${JSON.stringify(indices)}`);
      const idx = indices[0] >>> 0;
      const denom = Math.max(1, stepsEffective | 0);
      const gid = Math.floor(idx / denom);
      const off = idx % denom;
      const baseBig = bytesToBigIntBE(privs[gid]);
      let k = baseBig + BigInt(off);
      if (k >= SECP256K1_ORDER) k -= SECP256K1_ORDER;
      if (k === 0n) k = 1n;
      const kBytes = bigIntToBytesBE(k, 32);
      // Verify via walk (1 step) and compact
      try {
        const walkVerify = await vanity.encodeAndCommitWalkCompact([kBytes], 1, 0x0, 0);
        log(`walk indices: ${JSON.stringify(walkVerify.indices)}`);
      } catch (e) {
        log(`walk verify error: ${String(e)}`);
      }
      try {
        const compactVerify = await vanity.encodeAndCommitCompact([kBytes], 0x0, 0);
        log(`compact indices: ${JSON.stringify(compactVerify.indices)}`);
      } catch (e) {
        log(`compact verify error: ${String(e)}`);
      }
      log(`私钥: ${toHex(kBytes)}`);
      break;
    } else {
      log('No match in this batch');
    }
  }
}

ui.runMulti.addEventListener('click', () => {
  runMultiBatches().catch(err => {
    log(String(err));
    console.error(err);
  });
});



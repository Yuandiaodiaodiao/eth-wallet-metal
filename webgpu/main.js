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

async function runOneBatch() {
  logEl.textContent = '';
  const batchSize = Number.parseInt(ui.batchSize.value, 10) | 0;
  const stepsPerThread = Number.parseInt(ui.stepsPerThread.value, 10) | 0;
  const nibble = hexToInt(ui.nibble.value);
  const nibbleCount = Number.parseInt(ui.nibbleCount.value, 10) | 0;

  const totalThreads = batchSize;

  const { device } = await ensureWebGPU();
  const g16 = await WebGPUVanity.loadG16Buffer(device).catch(() => null);
  const vanity = new WebGPUVanity(device, g16);

  // Generate cryptographically strong random private keys
  const privs = generateRandomPrivKeys(totalThreads);

  const t0 = performance.now();
  const { indices, stepsEffective } = await vanity.encodeAndCommitWalkCompact(privs, stepsPerThread, nibble, nibbleCount);
  const t1 = performance.now();
  log(`avg rate: N/A | GPU: ${(t1 - t0).toFixed(2)} ms`);
  if (indices.length) log(`indices: ${JSON.stringify(indices)}`); else log('No match in this batch');
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
    const t0 = performance.now();
    const privs = generateRandomPrivKeys(totalThreads);
    const { indices } = await vanity.encodeAndCommitWalkCompact(privs, stepsPerThread, nibble, nibbleCount);
    const t1 = performance.now();
    totalKeys += totalThreads * stepsPerThread;
    const elapsed = Math.max((performance.now() - start) / 1000, 1e-6);
    const avgRate = totalKeys / elapsed;
    log(`batch: ${b}`);
    log(`avg rate: ${avgRate.toFixed(2)} keys/s (${(avgRate/1e6).toFixed(3)} MH/s) | GPU: ${(t1 - t0).toFixed(2)} ms`);
    if (indices.length) {
      log(`indices: ${JSON.stringify(indices)}`);
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



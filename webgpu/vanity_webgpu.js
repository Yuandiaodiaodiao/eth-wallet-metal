import { vanityWGSL } from './vanity_wgsl.js';

export class WebGPUVanity {
  constructor(device, g16Buffer) {
    this.device = device;
    this.g16Buffer = g16Buffer || null;
    this.module = device.createShaderModule({ code: vanityWGSL });
    this.pipelines = {
      compact: null,
      computeBase: null,
      walk: null,
    };
  }

  async ensurePipelines() {
    if (!this.pipelines.compact) {
      this.pipelines.compact = await this.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: this.module, entryPoint: 'vanity_kernel_w16_compact' },
      });
    }
    if (!this.pipelines.computeBase) {
      this.pipelines.computeBase = await this.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: this.module, entryPoint: 'vanity_kernel_compute_basepoint_w16' },
      });
    }
    if (!this.pipelines.walk) {
      this.pipelines.walk = await this.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: this.module, entryPoint: 'vanity_kernel_walk_compact' },
      });
    }
  }

  static async loadG16Buffer(device) {
    // 16 * 65536 * 64 bytes
    const resp = await fetch('./assets/g16_precomp_le.bin');
    if (!resp.ok) throw new Error('Failed to load g16_precomp_le.bin');
    const bin = new Uint8Array(await resp.arrayBuffer());
    const buf = device.createBuffer({ size: bin.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, bin);
    return buf;
  }

  // Encode and run single-shot compact pipeline (no walk)
  async encodeAndCommitCompact(privkeysBE32, nibble = 0x8, nibbleCount = 7) {
    await this.ensurePipelines();
    const count = privkeysBE32.length;
    if (count === 0) throw new Error('No keys');
    const inSize = 32 * count;
    const inBuf = this.device.createBuffer({ size: inSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const idxBuf = this.device.createBuffer({ size: 4 , usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cntBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // upload inputs
    {
      const joined = new Uint8Array(inSize);
      for (let i = 0; i < count; i++) joined.set(privkeysBE32[i], i * 32);
      this.device.queue.writeBuffer(inBuf, 0, joined);
      this.device.queue.writeBuffer(cntBuf, 0, new Uint32Array([0]));
    }

    // params: VanityParams {count, nibble, nibbleCount}
    const params = new Uint32Array([count >>> 0, (nibble & 0xf) >>> 0, nibbleCount >>> 0]);
    const pBuf = this.device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(pBuf, 0, params);

    const bind = this.device.createBindGroup({
      layout: this.pipelines.compact.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: cntBuf } },
        { binding: 3, resource: { buffer: pBuf } },
        { binding: 4, resource: { buffer: this.g16Buffer ?? this.device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }) } },
      ],
    });

    const workgroupSize = 256;
    const numWg = Math.ceil(count / workgroupSize);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.compact);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(numWg);
    pass.end();

    // readback
    const readIdx = this.device.createBuffer({ size: 4 , usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readCnt = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(idxBuf, 0, readIdx, 0, 4);
    enc.copyBufferToBuffer(cntBuf, 0, readCnt, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await readCnt.mapAsync(GPUMapMode.READ);
    const cnt = new Uint32Array(readCnt.getMappedRange())[0] >>> 0;
    readCnt.unmap();
    let indices = [];
    if (cnt > 0) {
      await readIdx.mapAsync(GPUMapMode.READ);
      const arr = new Uint32Array(readIdx.getMappedRange()).slice(0, cnt);
      indices = Array.from(arr);
      readIdx.unmap();
    }
    return { indices, stepsEffective: 1 };
  }

  // encode_and_commit_walk_compact equivalent (two-stage: compute base, walk)
  async encodeAndCommitWalkCompact(privkeysBE32, stepsPerThread = 8, nibble = 0x8, nibbleCount = 7) {
    console.log('111')
    await this.ensurePipelines();
    console.log('encodeAndCommitWalkCompact',privkeysBE32); 
    const count = privkeysBE32.length;
    const capacity = count * stepsPerThread;
    const inSize = 32 * count;
    const inBuf = this.device.createBuffer({ size: inSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const baseBuf = this.device.createBuffer({ size: count * 16 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const idxBuf = this.device.createBuffer({ size: 4 * 1, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cntBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    // upload inputs
    {
      const joined = new Uint8Array(inSize);
      for (let i = 0; i < count; i++) joined.set(privkeysBE32[i], i * 32);
      this.device.queue.writeBuffer(inBuf, 0, joined);
      this.device.queue.writeBuffer(cntBuf, 0, new Uint32Array([0]));
    }
    const params = new Uint32Array([count >>> 0, (nibble & 0xf) >>> 0, nibbleCount >>> 0, stepsPerThread >>> 0]);
    const pBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(pBuf, 0, params);

    // Stage 1: compute base
    const bind1 = this.device.createBindGroup({
      layout: this.pipelines.computeBase.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: baseBuf } },
        { binding: 2, resource: { buffer: pBuf } },
        { binding: 3, resource: { buffer: this.g16Buffer ?? this.device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE }) } },
      ],
    });

    // Stage 2: walk
    const bind2 = this.device.createBindGroup({
      layout: this.pipelines.walk.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: baseBuf } },
        { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: cntBuf } },
        { binding: 3, resource: { buffer: pBuf } },
      ],
    });

    const workgroupSize = 256;
    const numWg = Math.ceil(count / workgroupSize);
    const enc = this.device.createCommandEncoder();
    {
      const pass1 = enc.beginComputePass();
      pass1.setPipeline(this.pipelines.computeBase);
      pass1.setBindGroup(0, bind1);
      pass1.dispatchWorkgroups(numWg);
      pass1.end();
    }
    {
      const pass2 = enc.beginComputePass();
      pass2.setPipeline(this.pipelines.walk);
      pass2.setBindGroup(0, bind2);
      pass2.dispatchWorkgroups(numWg);
      pass2.end();
    }

    // readback
    const readIdx = this.device.createBuffer({ size: 4 , usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readCnt = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(idxBuf, 0, readIdx, 0, 4 );
    enc.copyBufferToBuffer(cntBuf, 0, readCnt, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await readCnt.mapAsync(GPUMapMode.READ);
    const cnt = new Uint32Array(readCnt.getMappedRange())[0] >>> 0;
    readCnt.unmap();
    let indices = [];
    if (cnt > 0) {
      await readIdx.mapAsync(GPUMapMode.READ);
      const arr = new Uint32Array(readIdx.getMappedRange()).slice(0, cnt);
      indices = Array.from(arr);
      readIdx.unmap();
    }
    return { indices, stepsEffective: stepsPerThread };
  }
}



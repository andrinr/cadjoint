/** GPU buffers the viewer rewrites in place as the drawn geometry changes. */

/**
 * A vertex buffer of fixed-stride records that grows on demand.
 *
 * Writing an empty array keeps the buffer and its contents and only zeroes
 * `count`, so every draw gates on `count` rather than on the buffer. A write
 * that does not fit replaces the buffer with one of at least 64 records.
 */
export class GrowableVertexBuffer {
  /** The GPU buffer, or null until the first non-empty write. */
  buffer: GPUBuffer | null = null;
  /** Records written by the last `write`. */
  count = 0;
  private capacity = 0;

  constructor(
    private readonly label: string,
    /** Bytes per record. */
    private readonly stride: number,
  ) {}

  write(device: GPUDevice, data: Float32Array<ArrayBuffer>): void {
    this.count = data.length / (this.stride / 4);
    if (data.length === 0) return;
    const required = data.byteLength;
    if (!this.buffer || this.capacity < required) {
      this.buffer?.destroy();
      const size = Math.max(required, this.stride * 64);
      this.buffer = device.createBuffer({
        label: this.label,
        size,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      this.capacity = size;
    }
    device.queue.writeBuffer(this.buffer, 0, data);
  }

  destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.capacity = 0;
    this.count = 0;
  }
}

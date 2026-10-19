/** The growth policy of the overlay's instance buffers, against a fake device. */

import { beforeAll, describe, expect, it } from "vitest";
import { GrowableVertexBuffer } from "../src/viewer/gpuBuffers";

interface FakeBuffer {
  label: string;
  size: number;
  destroyed: boolean;
  destroy(): void;
}

function fakeDevice() {
  const created: FakeBuffer[] = [];
  const writes: { buffer: FakeBuffer; bytes: number }[] = [];
  const device = {
    createBuffer({ label, size }: { label: string; size: number }) {
      const buffer: FakeBuffer = {
        label,
        size,
        destroyed: false,
        destroy() {
          buffer.destroyed = true;
        },
      };
      created.push(buffer);
      return buffer;
    },
    queue: {
      writeBuffer(buffer: FakeBuffer, _offset: number, data: Float32Array) {
        writes.push({ buffer, bytes: data.byteLength });
      },
    },
  };
  return { device: device as unknown as GPUDevice, created, writes };
}

beforeAll(() => {
  // Node has no WebGPU globals; the class reads only these two flags.
  Object.assign(globalThis, { GPUBufferUsage: { VERTEX: 32, COPY_DST: 8 } });
});

describe("GrowableVertexBuffer", () => {
  it("allocates at least 64 records and reuses the buffer while it fits", () => {
    const { device, created, writes } = fakeDevice();
    const instances = new GrowableVertexBuffer("edges", 16);
    instances.write(device, new Float32Array(8));
    expect(instances.count).toBe(2);
    expect(created).toHaveLength(1);
    expect(created[0].size).toBe(16 * 64);
    expect(created[0].label).toBe("edges");
    instances.write(device, new Float32Array(4 * 64));
    expect(created).toHaveLength(1);
    expect(writes).toHaveLength(2);
  });

  it("replaces a buffer that is too small", () => {
    const { device, created } = fakeDevice();
    const instances = new GrowableVertexBuffer("edges", 16);
    instances.write(device, new Float32Array(4));
    instances.write(device, new Float32Array(4 * 65));
    expect(created).toHaveLength(2);
    expect(created[0].destroyed).toBe(true);
    expect(created[1].size).toBe(16 * 65);
    expect(instances.count).toBe(65);
  });

  it("keeps the buffer on an empty write and only zeroes the count", () => {
    const { device, created, writes } = fakeDevice();
    const instances = new GrowableVertexBuffer("edges", 16);
    instances.write(device, new Float32Array(4));
    instances.write(device, new Float32Array(0));
    expect(instances.count).toBe(0);
    expect(instances.buffer).not.toBeNull();
    expect(created[0].destroyed).toBe(false);
    expect(writes).toHaveLength(1);
  });
});

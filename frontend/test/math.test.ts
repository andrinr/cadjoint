import { describe, expect, it } from "vitest";
import {
  cameraBasis,
  cameraPosition,
  intersectPlane,
  planeToWorld,
  projectPoint,
  rayFromPixel,
  viewProjection,
  worldToPlane,
  type Vec3,
} from "../src/viewer/math";

const POSITION: Vec3 = [4, 3, 6];
const TARGET: Vec3 = [0.3, -0.2, 0.1];
const WIDTH = 800;
const HEIGHT = 500;

/** Apply a column-major mat4 to a homogeneous point. */
function transform(matrix: Float32Array, point: Vec3): [number, number, number, number] {
  const out: number[] = [];
  for (let row = 0; row < 4; row++) {
    out.push(
      matrix[0 * 4 + row] * point[0] +
        matrix[1 * 4 + row] * point[1] +
        matrix[2 * 4 + row] * point[2] +
        matrix[3 * 4 + row],
    );
  }
  return out as [number, number, number, number];
}

describe("cameraPosition", () => {
  it("orbits at the requested distance from the target", () => {
    const camera = { yaw: 0.7, pitch: 0.3, distance: 5, target: [1, 2, 3] as Vec3 };
    const position = cameraPosition(camera);
    const delta = [
      position[0] - camera.target[0],
      position[1] - camera.target[1],
      position[2] - camera.target[2],
    ];
    expect(Math.hypot(...delta)).toBeCloseTo(5, 6);
  });
});

describe("cameraBasis", () => {
  it("produces an orthonormal right-handed frame", () => {
    const { forward, right, up } = cameraBasis(POSITION, TARGET);
    for (const axis of [forward, right, up]) {
      expect(Math.hypot(...axis)).toBeCloseTo(1, 6);
    }
    const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    expect(dot(forward, right)).toBeCloseTo(0, 6);
    expect(dot(forward, up)).toBeCloseTo(0, 6);
    expect(dot(right, up)).toBeCloseTo(0, 6);
  });
});

describe("projectPoint", () => {
  it("puts the camera target at the centre of the image", () => {
    const projected = projectPoint(TARGET, POSITION, TARGET, WIDTH, HEIGHT);
    expect(projected.visible).toBe(true);
    expect(projected.x).toBeCloseTo(WIDTH / 2, 4);
    expect(projected.y).toBeCloseTo(HEIGHT / 2, 4);
  });

  it("reports points behind the camera as not visible", () => {
    const behind: Vec3 = [8, 6, 12];
    expect(projectPoint(behind, POSITION, TARGET, WIDTH, HEIGHT).visible).toBe(false);
  });

  it("places a point offset along camera right to the right of centre", () => {
    const { right } = cameraBasis(POSITION, TARGET);
    const offset: Vec3 = [TARGET[0] + right[0], TARGET[1] + right[1], TARGET[2] + right[2]];
    const projected = projectPoint(offset, POSITION, TARGET, WIDTH, HEIGHT);
    expect(projected.x).toBeGreaterThan(WIDTH / 2);
  });

  it("places a point offset along camera up above centre", () => {
    const { up } = cameraBasis(POSITION, TARGET);
    const offset: Vec3 = [TARGET[0] + up[0], TARGET[1] + up[1], TARGET[2] + up[2]];
    const projected = projectPoint(offset, POSITION, TARGET, WIDTH, HEIGHT);
    // Framebuffer rows grow downward, so "up" means a smaller y.
    expect(projected.y).toBeLessThan(HEIGHT / 2);
  });
});

describe("rayFromPixel", () => {
  it("shoots along the forward axis through the image centre", () => {
    const { forward } = cameraBasis(POSITION, TARGET);
    const ray = rayFromPixel(WIDTH / 2, HEIGHT / 2, POSITION, TARGET, WIDTH, HEIGHT);
    expect(ray.direction[0]).toBeCloseTo(forward[0], 6);
    expect(ray.direction[1]).toBeCloseTo(forward[1], 6);
    expect(ray.direction[2]).toBeCloseTo(forward[2], 6);
  });

  it("round-trips with projectPoint for arbitrary pixels", () => {
    for (const [x, y] of [
      [10, 10],
      [640, 120],
      [399.5, 249.5],
      [790, 480],
    ]) {
      const ray = rayFromPixel(x, y, POSITION, TARGET, WIDTH, HEIGHT);
      const world: Vec3 = [
        ray.origin[0] + ray.direction[0] * 7,
        ray.origin[1] + ray.direction[1] * 7,
        ray.origin[2] + ray.direction[2] * 7,
      ];
      const projected = projectPoint(world, POSITION, TARGET, WIDTH, HEIGHT);
      expect(projected.x).toBeCloseTo(x, 3);
      expect(projected.y).toBeCloseTo(y, 3);
    }
  });
});

describe("viewProjection", () => {
  // The shader writes depth through this matrix while the overlay places its
  // vertices with it, and hit testing uses projectPoint — all three have to
  // agree or clicks land away from what is drawn.
  it("agrees with projectPoint on screen position", () => {
    const matrix = viewProjection(POSITION, TARGET, WIDTH / HEIGHT);
    for (const world of [
      [0, 0, 0],
      [1.5, -0.4, 0.8],
      [-2, 1, -1.5],
    ] as Vec3[]) {
      const clip = transform(matrix, world);
      const ndcX = clip[0] / clip[3];
      const ndcY = clip[1] / clip[3];
      const expected = projectPoint(world, POSITION, TARGET, WIDTH, HEIGHT);
      expect((ndcX * 0.5 + 0.5) * WIDTH).toBeCloseTo(expected.x, 3);
      expect((0.5 - ndcY * 0.5) * HEIGHT).toBeCloseTo(expected.y, 3);
    }
  });

  it("maps the near and far planes to 0 and 1", () => {
    const near = 0.05;
    const far = 200;
    const matrix = viewProjection(POSITION, TARGET, WIDTH / HEIGHT, near, far);
    const { forward } = cameraBasis(POSITION, TARGET);
    const at = (distance: number): Vec3 => [
      POSITION[0] + forward[0] * distance,
      POSITION[1] + forward[1] * distance,
      POSITION[2] + forward[2] * distance,
    ];
    const nearClip = transform(matrix, at(near));
    const farClip = transform(matrix, at(far));
    expect(nearClip[2] / nearClip[3]).toBeCloseTo(0, 5);
    expect(farClip[2] / farClip[3]).toBeCloseTo(1, 5);
  });

  it("increases depth monotonically away from the camera", () => {
    const matrix = viewProjection(POSITION, TARGET, WIDTH / HEIGHT);
    const { forward } = cameraBasis(POSITION, TARGET);
    let previous = -Infinity;
    for (const distance of [1, 2, 5, 10, 40]) {
      const clip = transform(matrix, [
        POSITION[0] + forward[0] * distance,
        POSITION[1] + forward[1] * distance,
        POSITION[2] + forward[2] * distance,
      ]);
      const depth = clip[2] / clip[3];
      expect(depth).toBeGreaterThan(previous);
      previous = depth;
    }
  });
});

describe("intersectPlane", () => {
  it("finds the hit point on the world XY plane", () => {
    const ray = { origin: [0, 0, 5] as Vec3, direction: [0, 0, -1] as Vec3 };
    const hit = intersectPlane(ray, [0, 0, 0], [0, 0, 1]);
    expect(hit).not.toBeNull();
    expect(hit![2]).toBeCloseTo(0, 6);
  });

  it("returns null for rays parallel to the plane", () => {
    const ray = { origin: [0, 1, 0] as Vec3, direction: [1, 0, 0] as Vec3 };
    expect(intersectPlane(ray, [0, 0, 0], [0, 1, 0])).toBeNull();
  });

  it("returns null when the plane is behind the ray", () => {
    const ray = { origin: [0, 0, 5] as Vec3, direction: [0, 0, 1] as Vec3 };
    expect(intersectPlane(ray, [0, 0, 0], [0, 0, 1])).toBeNull();
  });
});

describe("plane coordinates", () => {
  const origin: Vec3 = [0, 1, 0];
  const u: Vec3 = [0, 0, -1];
  const v: Vec3 = [0, 1, 0];

  it("round-trips sketch coordinates through world space", () => {
    const xy: [number, number] = [1.25, -0.5];
    const world = planeToWorld(xy, origin, u, v);
    const back = worldToPlane(world, origin, u, v);
    expect(back[0]).toBeCloseTo(xy[0], 6);
    expect(back[1]).toBeCloseTo(xy[1], 6);
  });

  it("recovers sketch coordinates from a picked ray hit", () => {
    // A click that lands on the plane should map back to the sketch point it
    // visually sits on — the basis of dragging a vertex.
    const planeNormal: Vec3 = [1, 0, 0];
    const ray = rayFromPixel(320, 210, POSITION, TARGET, WIDTH, HEIGHT);
    const hit = intersectPlane(ray, origin, planeNormal);
    expect(hit).not.toBeNull();
    const xy = worldToPlane(hit!, origin, u, v);
    const world = planeToWorld(xy, origin, u, v);
    expect(world[1]).toBeCloseTo(hit![1], 5);
    expect(world[2]).toBeCloseTo(hit![2], 5);
  });
});

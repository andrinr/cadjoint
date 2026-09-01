import { describe, expect, it } from "vitest";
import { sanitizeSceneName } from "../src/scenes";

describe("sanitizeSceneName", () => {
  it("keeps a plain .py name", () => {
    expect(sanitizeSceneName("bracket.py")).toBe("bracket.py");
    expect(sanitizeSceneName("part-2_v1.py")).toBe("part-2_v1.py");
  });

  it("appends the .py suffix when missing", () => {
    expect(sanitizeSceneName("bracket")).toBe("bracket.py");
    expect(sanitizeSceneName("  bracket  ")).toBe("bracket.py");
  });

  it("rejects path separators and traversal", () => {
    expect(sanitizeSceneName("../evil.py")).toBeNull();
    expect(sanitizeSceneName("..\\evil.py")).toBeNull();
    expect(sanitizeSceneName("nested/evil.py")).toBeNull();
    expect(sanitizeSceneName("/etc/passwd.py")).toBeNull();
  });

  it("rejects hidden files and empty stems", () => {
    expect(sanitizeSceneName(".hidden.py")).toBeNull();
    expect(sanitizeSceneName("..py")).toBeNull();
    expect(sanitizeSceneName("")).toBeNull();
    expect(sanitizeSceneName("   ")).toBeNull();
    expect(sanitizeSceneName(".py")).toBeNull();
  });

  it("rejects names beyond the length limit", () => {
    expect(sanitizeSceneName(`${"x".repeat(200)}.py`)).toBeNull();
  });
});

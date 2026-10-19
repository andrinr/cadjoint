import { describe, expect, it } from "vitest";
import { apiVersionMismatch, CLIENT_API_VERSION } from "../src/api";

describe("the API version check", () => {
  it("accepts a server that speaks this page's version", () => {
    expect(apiVersionMismatch(CLIENT_API_VERSION)).toBeNull();
  });

  it("names both versions when they differ", () => {
    const message = apiVersionMismatch(CLIENT_API_VERSION + 1);
    expect(message).toContain(`version ${CLIENT_API_VERSION}`);
    expect(message).toContain(`speaks version ${CLIENT_API_VERSION + 1}`);
    expect(message).toContain("npm run build");
  });
});

import { describe, it, expect } from "vitest";
import { DOMAIN_TOKENS, DEFAULT_PROFILES, DOMAIN_LABELS } from "@/theme/tokens";
import type { Domain } from "@/lib/api";

const DOMAINS: Domain[] = ["writing", "factory", "research"];

describe("DOMAIN_TOKENS", () => {
  it.each(DOMAINS)("%s has required CSS vars", (d) => {
    const t = DOMAIN_TOKENS[d];
    expect(t["--bg"]).toBeTruthy();
    expect(t["--accent"]).toBeTruthy();
    expect(t["--font-body"]).toBeTruthy();
  });
  it("factory background is dark", () => expect(DOMAIN_TOKENS.factory["--bg"]).toMatch(/^#0/));
  it("writing background is light", () => expect(DOMAIN_TOKENS.writing["--bg"]).toMatch(/^#F/i));
});

describe("DEFAULT_PROFILES", () => {
  it.each(DOMAINS)("%s profile has max_context_tokens", (d) => {
    expect(DEFAULT_PROFILES[d].max_context_tokens).toBeGreaterThan(0);
  });
  it("factory uses hard-interrupt", () =>
    expect(DEFAULT_PROFILES.factory.interaction_style).toBe("hard-interrupt"));
  it("research uses suggestive autonomy", () =>
    expect(DEFAULT_PROFILES.research.autonomy_level).toBe("suggestive"));
  it("research uses long horizon", () =>
    expect(DEFAULT_PROFILES.research.goal_horizon).toBe("long"));
});

describe("DOMAIN_LABELS", () => {
  it.each(DOMAINS)("%s has a label", (d) => {
    expect(typeof DOMAIN_LABELS[d]).toBe("string");
    expect(DOMAIN_LABELS[d].length).toBeGreaterThan(0);
  });
});

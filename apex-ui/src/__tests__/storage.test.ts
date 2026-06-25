import { describe, it, expect, beforeEach } from "vitest";
import {
  saveToken,
  getToken,
  clearToken,
  getCards,
  appendCard,
  getDismissed,
  addDismissed,
  clearUserData,
} from "@/lib/storage";
import type { ContextCard } from "@/lib/storage";

const UID = "user-abc";

function makeCard(n: number): ContextCard {
  return { chunk_id: `chunk-${n}`, text: `text ${n}`, ts: 1000 + n };
}

beforeEach(() => {
  localStorage.clear();
});

describe("token helpers", () => {
  it("getToken returns null when nothing stored", () => {
    expect(getToken()).toBeNull();
  });

  it("saveToken + getToken round-trip", () => {
    saveToken("tok-xyz");
    expect(getToken()).toBe("tok-xyz");
  });

  it("clearToken removes the token", () => {
    saveToken("tok-xyz");
    clearToken();
    expect(getToken()).toBeNull();
  });

  it("token key is not scoped to a user", () => {
    saveToken("tok-a");
    // still visible regardless of userId
    expect(getToken()).toBe("tok-a");
  });
});

describe("getCards / appendCard", () => {
  it("returns empty array when no cards stored", () => {
    expect(getCards(UID)).toEqual([]);
  });

  it("appends a card", () => {
    appendCard(UID, makeCard(1));
    expect(getCards(UID)).toHaveLength(1);
  });

  it("deduplicates by chunk_id", () => {
    appendCard(UID, makeCard(1));
    appendCard(UID, makeCard(1));
    expect(getCards(UID)).toHaveLength(1);
  });

  it("preserves order (append = newest last)", () => {
    appendCard(UID, makeCard(1));
    appendCard(UID, makeCard(2));
    const cards = getCards(UID);
    expect(cards[0].chunk_id).toBe("chunk-1");
    expect(cards[1].chunk_id).toBe("chunk-2");
  });

  it("cards for different users are isolated", () => {
    appendCard("user-a", makeCard(1));
    appendCard("user-b", makeCard(2));
    expect(getCards("user-a")).toHaveLength(1);
    expect(getCards("user-b")).toHaveLength(1);
    expect(getCards("user-a")[0].chunk_id).toBe("chunk-1");
  });

  it("trims to last 200 cards", () => {
    for (let i = 0; i < 205; i++) appendCard(UID, makeCard(i));
    expect(getCards(UID)).toHaveLength(200);
    // oldest entries are dropped
    expect(getCards(UID)[0].chunk_id).toBe("chunk-5");
  });

  it("handles corrupt localStorage gracefully", () => {
    localStorage.setItem("apex.user-abc.cards", "not-json{{");
    expect(getCards(UID)).toEqual([]);
  });
});

describe("getDismissed / addDismissed", () => {
  it("returns empty Set when nothing dismissed", () => {
    expect(getDismissed(UID).size).toBe(0);
  });

  it("addDismissed persists a chunk_id", () => {
    addDismissed(UID, "chunk-42");
    expect(getDismissed(UID).has("chunk-42")).toBe(true);
  });

  it("dismissed sets are isolated per user", () => {
    addDismissed("user-a", "chunk-1");
    addDismissed("user-b", "chunk-2");
    expect(getDismissed("user-a").has("chunk-1")).toBe(true);
    expect(getDismissed("user-a").has("chunk-2")).toBe(false);
  });

  it("handles corrupt localStorage gracefully", () => {
    localStorage.setItem("apex.user-abc.dismissed", "bad[json");
    expect(getDismissed(UID).size).toBe(0);
  });
});

describe("clearUserData", () => {
  it("removes cards and dismissed for that user", () => {
    appendCard(UID, makeCard(1));
    addDismissed(UID, "chunk-1");
    clearUserData(UID);
    expect(getCards(UID)).toEqual([]);
    expect(getDismissed(UID).size).toBe(0);
  });

  it("does not affect other users", () => {
    appendCard("user-other", makeCard(1));
    clearUserData(UID);
    expect(getCards("user-other")).toHaveLength(1);
  });
});

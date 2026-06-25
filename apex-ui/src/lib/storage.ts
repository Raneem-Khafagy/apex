/**
 * localStorage helpers — all keys scoped by user_id so multiple users
 * on the same browser never see each other's data.
 */

export interface ContextCard {
  chunk_id: string;
  text: string;
  ts: number;
  domain?: string;
}

const NS = "apex";
const MAX_CARDS = 200;

function key(userId: string, suffix: string): string {
  return `${NS}.${userId}.${suffix}`;
}

// ── Token / session ───────────────────────────────────────────────────────────

export function saveToken(token: string): void {
  localStorage.setItem(`${NS}.token`, token);
}

export function getToken(): string | null {
  return localStorage.getItem(`${NS}.token`);
}

export function clearToken(): void {
  localStorage.removeItem(`${NS}.token`);
}

// ── Per-user card history ─────────────────────────────────────────────────────

export function getCards(userId: string): ContextCard[] {
  try {
    return JSON.parse(
      localStorage.getItem(key(userId, "cards")) ?? "[]"
    ) as ContextCard[];
  } catch {
    return [];
  }
}

export function appendCard(userId: string, card: ContextCard): void {
  const cards = getCards(userId);
  if (cards.find((c) => c.chunk_id === card.chunk_id)) return; // dedup
  const trimmed = [...cards, card].slice(-MAX_CARDS);
  localStorage.setItem(key(userId, "cards"), JSON.stringify(trimmed));
}

// ── Per-user dismissed set ────────────────────────────────────────────────────

export function getDismissed(userId: string): Set<string> {
  try {
    return new Set(
      JSON.parse(localStorage.getItem(key(userId, "dismissed")) ?? "[]") as string[]
    );
  } catch {
    return new Set();
  }
}

export function addDismissed(userId: string, chunkId: string): void {
  const s = getDismissed(userId);
  s.add(chunkId);
  localStorage.setItem(key(userId, "dismissed"), JSON.stringify([...s]));
}

export function clearUserData(userId: string): void {
  localStorage.removeItem(key(userId, "cards"));
  localStorage.removeItem(key(userId, "dismissed"));
}

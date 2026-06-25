/**
 * Typed fetch wrappers for all APEX server endpoints.
 * Auth token is read from localStorage and injected automatically.
 */

/** Domain is a free-form string — users define their own. */
export type Domain = string;

// ── Auth ─────────────────────────────────────────────────────────────────────

export interface AuthResponse {
  token: string;
  user_id: string;
  username: string;
  domain: Domain;
  onboarded: boolean;
}

export interface MeResponse {
  user_id: string;
  username: string;
  domain: Domain;
  subscriber_id: string;
  onboarded: boolean;
  profile_json: string;
}

export interface OnboardRequest {
  domain: Domain;
  profile: Partial<ConsumerProfile>;
}

export interface ConsumerProfile {
  autonomy_level: string;
  goal_horizon: string;
  interaction_style: string;
  output_format: string;
  vocabulary_level: string;
  verbosity: string;
  citation_style: string;
  max_context_tokens: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getToken(): string | null {
  return localStorage.getItem("apex.token");
}

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  auth = false
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const detail = await res.text();
    throw Object.assign(new Error(detail), { status: res.status });
  }
  return res.json() as Promise<T>;
}

// ── Auth endpoints ────────────────────────────────────────────────────────────

export function register(
  username: string,
  password: string,
  domain: Domain
): Promise<AuthResponse> {
  return apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password, domain }),
  });
}

export function login(
  username: string,
  password: string
): Promise<AuthResponse> {
  return apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function getMe(): Promise<MeResponse> {
  return apiFetch("/auth/me", {}, true);
}

export function onboard(req: OnboardRequest): Promise<{ subscriber_id: string; domain: Domain }> {
  return apiFetch("/auth/onboard", {
    method: "POST",
    body: JSON.stringify(req),
  }, true);
}

// ── Vault / knowledge-base config ────────────────────────────────────────────

export interface VaultConfig {
  vault_path: string;
  index_path: string;
  doc_count: number;
  reindexing: boolean;
}

export function getVaultConfig(): Promise<VaultConfig> {
  return apiFetch("/config/vault", {}, true);
}

export function setVaultPath(vault_path: string): Promise<{ vault_path: string; status: string }> {
  return apiFetch("/config/vault", {
    method: "POST",
    body: JSON.stringify({ vault_path, fresh: false }),
  }, true);
}

export function startFreshVault(): Promise<{ vault_path: string; status: string }> {
  return apiFetch("/config/vault", {
    method: "POST",
    body: JSON.stringify({ fresh: true }),
  }, true);
}

export function triggerReindex(): Promise<{ status: string; vault_path: string }> {
  return apiFetch("/config/reindex", { method: "POST" }, true);
}

// ── MCP endpoints ─────────────────────────────────────────────────────────────

export function getContext(subscriberId: string): Promise<{ subscriber_id: string; context: string }> {
  return apiFetch(`/context/${subscriberId}`, {}, true);
}

export function wsStreamUrl(subscriberId: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/stream/${subscriberId}?format=json`;
}

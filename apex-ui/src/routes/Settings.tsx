/**
 * Settings — ConsumerProfile editor + domain change for the logged-in user.
 */
import React, { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { onboard, getVaultConfig, setVaultPath, startFreshVault, triggerReindex } from "@/lib/api";
import type { ConsumerProfile, VaultConfig } from "@/lib/api";
import { getDefaultProfile, getDomainIcon, getDomainLabel } from "@/theme/tokens";

const DOMAIN_SUGGESTIONS = [
  "writing", "research", "factory", "medical", "legal",
  "engineering", "teaching", "logistics", "finance", "security",
];

export default function Settings() {
  const { user, refreshMe } = useAuth();

  const initialProfile = (): ConsumerProfile => {
    if (!user) return getDefaultProfile("general");
    try {
      return { ...getDefaultProfile(user.domain), ...JSON.parse(user.profile_json) };
    } catch {
      return getDefaultProfile(user.domain);
    }
  };

  const [domain, setDomain]   = useState(user?.domain ?? "general");
  const [profile, setProfile] = useState<ConsumerProfile>(initialProfile);
  const [saving, setSaving]   = useState(false);
  const [saved, setSaved]     = useState(false);
  const [error, setError]     = useState("");

  // ── Vault / knowledge-base state ─────────────────────────────────────────
  const [vault, setVault]             = useState<VaultConfig | null>(null);
  const [vaultInput, setVaultInput]   = useState("");
  const [vaultStatus, setVaultStatus] = useState("");
  const [vaultError, setVaultError]   = useState("");

  const loadVault = useCallback(async () => {
    try {
      const v = await getVaultConfig();
      setVault(v);
      setVaultInput(v.vault_path);
    } catch { /* daemon may not be running */ }
  }, []);

  useEffect(() => { loadVault(); }, [loadVault]);

  // Poll while reindexing
  useEffect(() => {
    if (!vault?.reindexing) return;
    const id = setInterval(async () => {
      const v = await getVaultConfig().catch(() => null);
      if (v) { setVault(v); if (!v.reindexing) clearInterval(id); }
    }, 2000);
    return () => clearInterval(id);
  }, [vault?.reindexing]);

  async function handleSetVault() {
    setVaultError(""); setVaultStatus("");
    try {
      const res = await setVaultPath(vaultInput);
      setVaultStatus(res.status);
      await loadVault();
    } catch (e) {
      setVaultError(e instanceof Error ? e.message : "Failed to set path");
    }
  }

  async function handleFresh() {
    setVaultError(""); setVaultStatus("");
    try {
      const res = await startFreshVault();
      setVaultInput(res.vault_path);
      setVaultStatus(res.status);
      await loadVault();
    } catch (e) {
      setVaultError(e instanceof Error ? e.message : "Failed");
    }
  }

  async function handleReindex() {
    setVaultError(""); setVaultStatus("");
    try {
      await triggerReindex();
      setVaultStatus("Re-indexing started…");
      await loadVault();
    } catch (e) {
      setVaultError(e instanceof Error ? e.message : "Failed to start reindex");
    }
  }

  function update<K extends keyof ConsumerProfile>(k: K, v: ConsumerProfile[K]) {
    setProfile((p) => ({ ...p, [k]: v }));
  }

  async function handleSave() {
    setSaving(true);
    setError("");
    try {
      const d = domain.trim().toLowerCase() || "general";
      await onboard({ domain: d, profile });
      await refreshMe();
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const labelStyle = { color: "var(--text-muted)", fontSize: 12, fontWeight: 500 };

  const seg = (label: string, options: string[], val: string, onChange: (v: string) => void) => (
    <div className="mb-4">
      <label className="block mb-1" style={labelStyle}>{label}</label>
      <div className="flex flex-wrap gap-1">
        {options.map((o) => (
          <button key={o} type="button" onClick={() => onChange(o)}
            className="px-3 py-1 rounded text-xs transition-colors"
            style={{
              background: val === o ? "var(--accent)" : "var(--surface-2)",
              color: val === o ? "#fff" : "var(--text-muted)",
              border: "1px solid",
              borderColor: val === o ? "var(--accent)" : "var(--border)",
            }}>{o}</button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex h-full" style={{ background: "var(--bg)" }}>
      <div className="flex flex-col flex-1 overflow-hidden">

        {/* Header */}
        <div className="flex items-center px-5 py-3 border-b shrink-0"
          style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <h1 className="text-sm font-semibold" style={{ color: "var(--text)" }}>Settings</h1>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5 max-w-xl">

          {/* Knowledge Base */}
          <section className="mb-8">
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3"
              style={{ color: "var(--text-muted)" }}>Knowledge Base</h2>

            <div className="p-4 rounded border mb-3"
              style={{ background: "var(--surface)", borderColor: "var(--border)" }}>

              {/* Current status */}
              <div className="flex items-center gap-3 mb-3">
                <span style={{ fontSize: 20 }}>🗄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate" style={{ color: "var(--text)" }}>
                    {vault ? vault.vault_path : "Loading…"}
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                    {vault?.reindexing
                      ? "⟳ Re-indexing…"
                      : vault
                        ? `${vault.doc_count} chunk${vault.doc_count !== 1 ? "s" : ""} indexed`
                        : "Daemon not running"}
                  </p>
                </div>
              </div>

              {/* Path input */}
              <label className="block mb-1" style={labelStyle}>Folder path on this machine</label>
              <div className="flex gap-2 mb-2">
                <input
                  className="flex-1 px-3 py-2 rounded text-sm border outline-none"
                  style={{
                    background: "var(--surface-2)",
                    borderColor: "var(--border)",
                    color: "var(--text)",
                  }}
                  placeholder="/Users/you/Documents/MyVault"
                  value={vaultInput}
                  onChange={(e) => setVaultInput(e.target.value)}
                  spellCheck={false}
                />
                <button onClick={handleSetVault}
                  className="px-3 py-2 rounded text-xs font-medium shrink-0"
                  style={{ background: "var(--accent)", color: "#fff" }}>
                  Set path
                </button>
              </div>
              <p className="text-xs mb-3" style={{ color: "var(--text-muted)" }}>
                Type the absolute path to a folder of documents (markdown, txt, pdf) on your machine.
                APEX will monitor this folder and retrieve from it.
              </p>

              {/* Action buttons */}
              <div className="flex gap-2">
                <button onClick={handleReindex} disabled={vault?.reindexing}
                  className="flex-1 py-1.5 rounded text-xs font-medium disabled:opacity-40 transition-opacity"
                  style={{ background: "var(--surface-2)", color: "var(--text)", border: "1px solid var(--border)" }}>
                  {vault?.reindexing ? "⟳ Indexing…" : "↻ Re-index now"}
                </button>
                <button onClick={handleFresh}
                  className="flex-1 py-1.5 rounded text-xs font-medium transition-opacity"
                  style={{ background: "var(--surface-2)", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
                  ✕ Start fresh
                </button>
              </div>

              {vaultStatus && (
                <p className="text-xs mt-2 px-2 py-1.5 rounded"
                  style={{ background: "var(--accent-dim)", color: "var(--accent)" }}>
                  {vaultStatus}
                </p>
              )}
              {vaultError && (
                <p className="text-xs mt-2 px-2 py-1.5 rounded"
                  style={{ background: "#2a1515", color: "#f87171" }}>
                  {vaultError}
                </p>
              )}
            </div>
          </section>

          {/* Domain */}
          <section className="mb-6">
            <h2 className="text-xs font-bold uppercase tracking-widest mb-3"
              style={{ color: "var(--text-muted)" }}>Domain</h2>

            <div className="flex items-center gap-2 mb-2 px-3 py-2 rounded"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
              <span style={{ fontSize: 18 }}>{getDomainIcon(domain)}</span>
              <span className="text-sm font-medium" style={{ color: "var(--text)" }}>
                {getDomainLabel(domain)}
              </span>
            </div>

            <label className="block mb-1" style={labelStyle}>Change domain</label>
            <input
              className="w-full px-3 py-2 rounded text-sm border outline-none focus:ring-2"
              style={{
                background: "var(--surface-2)",
                borderColor: "var(--border)",
                color: "var(--text)",
                outlineColor: "var(--accent)",
              }}
              placeholder="e.g. medical, legal, robotics…"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              list="settings-domain-suggestions"
            />
            <datalist id="settings-domain-suggestions">
              {DOMAIN_SUGGESTIONS.map((s) => <option key={s} value={s} />)}
            </datalist>
            <div className="flex flex-wrap gap-1 mt-2">
              {DOMAIN_SUGGESTIONS.slice(0, 6).map((s) => (
                <button key={s} type="button"
                  onClick={() => setDomain(s)}
                  className="px-2 py-0.5 rounded text-xs transition-colors"
                  style={{
                    background: domain === s ? "var(--accent)" : "var(--surface-2)",
                    color: domain === s ? "#fff" : "var(--text-muted)",
                    border: "1px solid",
                    borderColor: domain === s ? "var(--accent)" : "var(--border)",
                  }}>{s}</button>
              ))}
            </div>
          </section>

          {/* Consumer Profile */}
          <section className="mb-6 p-4 rounded border"
            style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <h2 className="text-xs font-bold uppercase tracking-widest mb-4"
              style={{ color: "var(--text-muted)" }}>Consumer Profile</h2>

            {seg("Autonomy level", ["suggestive", "assistive", "autonomous"],
              profile.autonomy_level, (v) => update("autonomy_level", v))}
            {seg("Goal horizon", ["short", "mid", "long"],
              profile.goal_horizon, (v) => update("goal_horizon", v))}
            {seg("Interaction style", ["ambient", "soft-interrupt", "hard-interrupt", "conversational"],
              profile.interaction_style, (v) => update("interaction_style", v))}
            {seg("Vocabulary level", ["technical", "domain-expert", "layman"],
              profile.vocabulary_level, (v) => update("vocabulary_level", v))}
            {seg("Verbosity", ["concise", "standard", "detailed"],
              profile.verbosity, (v) => update("verbosity", v))}
            {seg("Citation style", ["inline", "footnote", "none"],
              profile.citation_style, (v) => update("citation_style", v))}

            <div className="mb-4">
              <label className="block mb-1" style={labelStyle}>
                Max context tokens: {profile.max_context_tokens}
              </label>
              <input type="range" min={128} max={2048} step={64}
                value={profile.max_context_tokens}
                onChange={(e) => update("max_context_tokens", Number(e.target.value))}
                className="w-full" style={{ accentColor: "var(--accent)" }} />
            </div>
          </section>

          {error && (
            <p className="text-xs mb-3 px-3 py-2 rounded"
              style={{ background: "#2a1515", color: "#f87171" }}>{error}</p>
          )}

          <button onClick={handleSave} disabled={saving}
            className="px-5 py-2 rounded font-semibold text-sm disabled:opacity-50"
            style={{ background: "var(--accent)", color: "#fff" }}>
            {saving ? "Saving…" : saved ? "Saved ✓" : "Save & re-subscribe"}
          </button>
        </div>
      </div>
    </div>
  );
}

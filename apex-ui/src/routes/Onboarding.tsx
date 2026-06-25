/**
 * Onboarding — ConsumerProfile config after first register.
 * Domain is already set (chosen at registration). Here the user
 * configures how APEX should interact with them.
 */
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { onboard } from "@/lib/api";
import type { ConsumerProfile } from "@/lib/api";
import { getDomainIcon, getDomainLabel, getDefaultProfile } from "@/theme/tokens";

export default function Onboarding() {
  const { user, refreshMe } = useAuth();
  const navigate = useNavigate();

  const domain  = user?.domain ?? "general";
  const [profile, setProfile] = useState<ConsumerProfile>(getDefaultProfile(domain));
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  function update<K extends keyof ConsumerProfile>(k: K, v: ConsumerProfile[K]) {
    setProfile((p) => ({ ...p, [k]: v }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await onboard({ domain, profile });
      await refreshMe();
      navigate("/stream", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Setup failed");
    } finally {
      setLoading(false);
    }
  }

  const seg = (
    label: string,
    options: string[],
    value: string,
    onChange: (v: string) => void
  ) => (
    <div className="mb-4">
      <label className="block text-xs font-medium mb-1" style={{ color: "#9ca3af" }}>
        {label}
      </label>
      <div className="flex flex-wrap gap-1">
        {options.map((o) => (
          <button key={o} type="button" onClick={() => onChange(o)}
            className="px-3 py-1 rounded text-xs transition-colors"
            style={{
              background: value === o ? "#6366f1" : "#1f2937",
              color: value === o ? "#fff" : "#9ca3af",
              border: "1px solid",
              borderColor: value === o ? "#6366f1" : "#374151",
            }}>
            {o}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "#0d1117" }}>
      <div className="w-full max-w-lg">

        <div className="flex flex-col items-center mb-6">
          <span style={{ fontSize: 32, color: "#6366f1" }}>⬡</span>
          <h1 className="text-xl font-bold text-white mt-1">Set up your profile</h1>
          <p className="text-xs mt-1" style={{ color: "#9ca3af" }}>
            APEX will adapt its proactive context to your domain
          </p>
        </div>

        <div className="rounded-xl border px-8 py-6"
          style={{ background: "#161b22", borderColor: "#30363d" }}>

          {/* Domain identity banner */}
          <div className="flex items-center gap-3 mb-6 px-4 py-3 rounded-lg"
            style={{ background: "#1f2937", border: "1px solid #374151" }}>
            <span style={{ fontSize: 24 }}>{getDomainIcon(domain)}</span>
            <div>
              <p className="text-xs" style={{ color: "#9ca3af" }}>Your domain</p>
              <p className="text-sm font-semibold" style={{ color: "#e0e7ff" }}>
                {getDomainLabel(domain)}
              </p>
            </div>
            <p className="ml-auto text-xs" style={{ color: "#6b7280" }}>
              (change in Settings later)
            </p>
          </div>

          <hr style={{ borderColor: "#30363d", marginBottom: 20 }} />

          <form onSubmit={handleSubmit}>
            <p className="text-xs font-bold uppercase tracking-widest mb-4"
              style={{ color: "#6b7280" }}>How should APEX interact with you?</p>

            {seg("Autonomy level", ["suggestive", "assistive", "autonomous"],
              profile.autonomy_level, (v) => update("autonomy_level", v))}
            {seg("Interaction style", ["ambient", "soft-interrupt", "hard-interrupt", "conversational"],
              profile.interaction_style, (v) => update("interaction_style", v))}
            {seg("Verbosity", ["concise", "standard", "detailed"],
              profile.verbosity, (v) => update("verbosity", v))}

            <div className="mb-6">
              <label className="block text-xs font-medium mb-1" style={{ color: "#9ca3af" }}>
                Max context tokens: {profile.max_context_tokens}
              </label>
              <input type="range" min={128} max={2048} step={64}
                value={profile.max_context_tokens}
                onChange={(e) => update("max_context_tokens", Number(e.target.value))}
                className="w-full" style={{ accentColor: "#6366f1" }} />
              <div className="flex justify-between text-xs mt-0.5" style={{ color: "#6b7280" }}>
                <span>128 — very concise</span>
                <span>2048 — full context</span>
              </div>
            </div>

            {error && (
              <p className="text-xs mb-3 px-3 py-2 rounded"
                style={{ background: "#2a1515", color: "#f87171" }}>{error}</p>
            )}

            <button type="submit" disabled={loading}
              className="w-full py-2 rounded font-semibold text-sm disabled:opacity-50"
              style={{ background: "#6366f1", color: "#fff" }}>
              {loading ? "Setting up…" : "Start using APEX →"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

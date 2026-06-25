/**
 * Login — login + register on the same page (tabbed).
 * Domain is free-form: the user types whatever describes their context.
 */
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";

type Tab = "login" | "register";

const DOMAIN_SUGGESTIONS = [
  "writing", "research", "factory", "medical", "legal",
  "engineering", "teaching", "logistics", "finance", "security",
];

export default function Login() {
  const [tab, setTab]           = useState<Tab>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [domain, setDomain]     = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const { login, register, user } = useAuth();
  const navigate = useNavigate();

  if (user) {
    navigate(user.onboarded ? "/stream" : "/onboarding", { replace: true });
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (tab === "login") {
        await login(username, password);
      } else {
        const d = domain.trim().toLowerCase() || "general";
        await register(username, password, d);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Request failed";
      try { setError(JSON.parse(msg).detail ?? msg); }
      catch { setError(msg); }
    } finally {
      setLoading(false);
    }
  }

  const input =
    "w-full px-3 py-2 rounded text-sm border outline-none focus:ring-2 " +
    "bg-gray-800 border-gray-700 text-gray-100 placeholder-gray-500 focus:ring-indigo-500";

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "#0d1117" }}>
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8 gap-1">
          <span style={{ fontSize: 36, color: "#6366f1" }}>⬡</span>
          <h1 className="text-2xl font-bold text-white tracking-wide">APEX</h1>
          <p className="text-xs" style={{ color: "#9ca3af" }}>
            Proactive AI · On-device · Adapts to you
          </p>
        </div>

        {/* Card */}
        <div className="rounded-xl border px-8 py-7"
          style={{ background: "#161b22", borderColor: "#30363d" }}>

          {/* Tabs */}
          <div className="flex gap-1 mb-6 border-b" style={{ borderColor: "#30363d" }}>
            {(["login", "register"] as Tab[]).map((t) => (
              <button key={t} onClick={() => { setTab(t); setError(""); }}
                className="px-4 pb-2 text-sm font-medium capitalize transition-colors"
                style={{
                  color: tab === t ? "#e0e7ff" : "#9ca3af",
                  borderBottom: tab === t ? "2px solid #6366f1" : "2px solid transparent",
                }}>
                {t}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-3">
            <input className={input} placeholder="Username"
              value={username} onChange={(e) => setUsername(e.target.value)}
              autoComplete="username" required />

            <input type="password" className={input}
              placeholder="Password (min 6 chars)"
              value={password} onChange={(e) => setPassword(e.target.value)}
              autoComplete={tab === "login" ? "current-password" : "new-password"}
              required />

            {tab === "register" && (
              <div>
                <label className="block text-xs mb-1" style={{ color: "#9ca3af" }}>
                  What is your domain? <span style={{ color: "#6b7280" }}>(anything)</span>
                </label>
                <input className={input}
                  placeholder="e.g. medical, legal, robotics, logistics…"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  autoComplete="off"
                  list="domain-suggestions"
                />
                <datalist id="domain-suggestions">
                  {DOMAIN_SUGGESTIONS.map((s) => <option key={s} value={s} />)}
                </datalist>
                {/* Quick-pick chips */}
                <div className="flex flex-wrap gap-1 mt-2">
                  {DOMAIN_SUGGESTIONS.slice(0, 6).map((s) => (
                    <button key={s} type="button"
                      onClick={() => setDomain(s)}
                      className="px-2 py-0.5 rounded text-xs transition-colors"
                      style={{
                        background: domain === s ? "#6366f1" : "#1f2937",
                        color: domain === s ? "#fff" : "#6b7280",
                        border: "1px solid",
                        borderColor: domain === s ? "#6366f1" : "#374151",
                      }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <p className="text-xs rounded px-3 py-2"
                style={{ background: "#2a1515", color: "#f87171" }}>{error}</p>
            )}

            <button type="submit" disabled={loading}
              className="w-full py-2 rounded font-semibold text-sm mt-1 disabled:opacity-50 transition-opacity"
              style={{ background: "#6366f1", color: "#fff" }}>
              {loading ? "…" : tab === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

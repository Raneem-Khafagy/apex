/** MetricsOverlay — 28 px bottom strip, revealed by Cmd/Ctrl+Shift+R */
import React from "react";
import { useApex } from "@/contexts/ApexContext";

export function MetricsOverlay() {
  const { metrics, buffer, pipeline } = useApex();
  const total = Object.values(buffer).reduce((s, v) => s + (v ?? 0), 0);
  const { prp, ltc, dps } = metrics;
  const tau = pipeline.tau;

  const col = (ok: boolean | undefined) =>
    ok == null ? "#8b949e" : ok ? "#3fb950" : "#f85149";

  return (
    <div
      className="flex items-center gap-5 px-4 text-xs border-t shrink-0"
      style={{
        height: 28,
        background: "#0d1117",
        color: "#c9d1d9",
        borderColor: "#30363d",
        fontFamily: "'SF Mono','Fira Code',monospace",
        zIndex: 50,
      }}
    >
      <span>PRP: <strong style={{ color: col(prp != null ? prp >= 0.65 : undefined) }}>{prp != null ? prp.toFixed(2) : "—"}</strong></span>
      <span>LtC: <strong style={{ color: col(ltc != null ? ltc < 0 : undefined) }}>{ltc != null ? `${Math.round(ltc).toLocaleString()} ms` : "—"}</strong></span>
      <span>DPS: <strong style={{ color: col(dps != null ? dps >= 0.75 : undefined) }}>{dps != null ? dps.toFixed(2) : "—"}</strong></span>
      <span>buf: <strong style={{ color: "#58a6ff" }}>{total}</strong></span>
      <span>τ: <strong>{tau != null ? tau.toFixed(2) : "—"}</strong></span>
      <span style={{ color: "#8b949e", marginLeft: "auto" }}>Researcher mode · ⌘⇧R to hide</span>
    </div>
  );
}

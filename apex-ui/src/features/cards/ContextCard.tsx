/**
 * ContextCard — domain-skinned card in the stream feed.
 * Save triggers a pull-claim (thesis evaluation event).
 * Dismiss is client-side only.
 */
import React from "react";
import clsx from "clsx";
import type { ContextCard as Card } from "@/lib/storage";
import type { Domain } from "@/lib/api";

interface Props {
  card: Card;
  domain: Domain;
  dismissed?: boolean;
  onDismiss: (id: string) => void;
  onSave?: (id: string) => void;
}

const LABEL: Record<Domain, string> = {
  writing:  "Context",
  factory:  "APEX ALERT",
  research: "Related work",
};

export function ContextCard({ card, domain, dismissed, onDismiss, onSave }: Props) {
  if (dismissed) return null;

  const isFactory = domain === "factory";
  const ts = new Date(card.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div
      className={clsx(
        "group relative mb-3 p-4 border text-sm transition-shadow hover:shadow-md",
        isFactory && "border-l-4"
      )}
      style={{
        background: "var(--surface)",
        borderColor: isFactory ? "var(--danger)" : "var(--border)",
        borderRadius: "var(--card-radius)",
        fontFamily: "var(--font-body)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span
          className="text-xs font-bold uppercase tracking-widest"
          style={{ color: isFactory ? "var(--danger)" : "var(--accent)" }}
        >
          {LABEL[domain]}
        </span>
        <span className="text-xs" style={{ color: "var(--text-muted)" }}>{ts}</span>
      </div>

      {/* Body */}
      <p
        className="leading-relaxed whitespace-pre-wrap"
        style={{ color: "var(--text)" }}
      >
        {card.text.length > 500 ? card.text.slice(0, 500) + "…" : card.text}
      </p>

      {/* Actions */}
      <div className="flex gap-2 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
        {onSave && (
          <button
            onClick={() => onSave(card.chunk_id)}
            className="text-xs px-3 py-1 rounded font-medium"
            style={{ background: "var(--accent-dim)", color: "var(--accent)" }}
          >
            {isFactory ? "Acknowledge" : "Claim"}
          </button>
        )}
        <button
          onClick={() => onDismiss(card.chunk_id)}
          className="text-xs px-3 py-1 rounded"
          style={{ background: "var(--surface-2)", color: "var(--text-muted)" }}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

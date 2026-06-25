/**
 * CardFeed — chronological stream of APEX context cards.
 * Newest cards appear at the top (reverse chronological).
 * "Pull context" button manually claims context (evaluation event).
 */
import React, { useState, useCallback, useEffect } from "react";
import clsx from "clsx";
import type { Domain } from "@/lib/api";
import { getContext } from "@/lib/api";
import { useStreamWS } from "@/hooks/useStreamWS";
import {
  getCards,
  appendCard,
  getDismissed,
  addDismissed,
  type ContextCard,
} from "@/lib/storage";
import { ContextCard as CardComponent } from "./ContextCard";
import { useApex } from "@/contexts/ApexContext";

interface Props {
  userId: string;
  subscriberId: string;
  domain: Domain;
}

export function CardFeed({ userId, subscriberId, domain }: Props) {
  const [cards, setCards] = useState<ContextCard[]>(() => getCards(userId));
  const [dismissed, setDismissed] = useState<Set<string>>(
    () => getDismissed(userId)
  );
  const [pulling, setPulling] = useState(false);
  const { signal, pipeline } = useApex();

  const handleNewCard = useCallback(
    (card: ContextCard) => {
      const c = { ...card, domain };
      setCards((prev) => {
        if (prev.find((x) => x.chunk_id === c.chunk_id)) return prev;
        return [c, ...prev]; // newest first
      });
      appendCard(userId, c);
    },
    [userId, domain]
  );

  useStreamWS({ subscriberId, onCard: handleNewCard });

  const handleDismiss = useCallback(
    (id: string) => {
      setDismissed((prev) => {
        const next = new Set(prev);
        next.add(id);
        addDismissed(userId, id);
        return next;
      });
    },
    [userId]
  );

  const handlePull = useCallback(async () => {
    setPulling(true);
    try {
      const res = await getContext(subscriberId);
      if (res.context) {
        handleNewCard({
          chunk_id: crypto.randomUUID(),
          text: res.context,
          ts: Date.now() / 1000,
          domain,
        });
      }
    } finally {
      setPulling(false);
    }
  }, [subscriberId, domain, handleNewCard]);

  const visible = cards.filter((c) => !dismissed.has(c.chunk_id));

  // Inline indicator: confidence crossed tau
  const conf = signal.confidence ?? 0;
  const tau = pipeline.tau ?? 0.65;
  const warm = conf >= tau;

  return (
    <div className="flex flex-col h-full">
      {/* Pull bar */}
      <div
        className="flex items-center gap-3 px-4 py-3 border-b shrink-0"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <button
          onClick={handlePull}
          disabled={pulling}
          className="text-sm px-4 py-1.5 rounded font-medium disabled:opacity-50 transition-opacity"
          style={{ background: "var(--accent)", color: "#fff" }}
        >
          {pulling ? "Fetching…" : "Pull context"}
        </button>

        {/* Layer 2 inline indicator */}
        {warm && (
          <span
            className="apex-pulse flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full"
            style={{
              background: "var(--accent-dim)",
              border: "1px solid var(--accent)",
              color: "var(--accent)",
            }}
          >
            <span
              style={{
                width: 6, height: 6, borderRadius: "50%",
                background: "var(--accent)", display: "inline-block",
              }}
            />
            APEX ready · c={conf.toFixed(2)}
          </span>
        )}

        <span className="ml-auto text-xs" style={{ color: "var(--text-muted)" }}>
          {visible.length} card{visible.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {visible.length === 0 ? (
          <div
            className="flex flex-col items-center justify-center h-full gap-3"
            style={{ color: "var(--text-muted)" }}
          >
            <span className="text-4xl opacity-30">⬡</span>
            <p className="text-sm text-center max-w-xs">
              APEX is watching your activity.
              <br />
              Context will appear here proactively — before you ask.
            </p>
          </div>
        ) : (
          visible.map((card) => (
            <CardComponent
              key={card.chunk_id}
              card={card}
              domain={domain}
              onDismiss={handleDismiss}
              onSave={handlePull}
            />
          ))
        )}
      </div>
    </div>
  );
}

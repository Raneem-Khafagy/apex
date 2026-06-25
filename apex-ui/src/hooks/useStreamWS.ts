/**
 * useStreamWS — WebSocket connection to /stream/{subscriber_id}?format=json.
 * Exponential back-off reconnect: 1s → 2s → 4s → … → 30s max.
 */
import { useEffect, useRef } from "react";
import { wsStreamUrl } from "@/lib/api";
import type { ContextCard } from "@/lib/storage";

interface Opts {
  subscriberId: string | null;
  onCard: (card: ContextCard) => void;
}

export function useStreamWS({ subscriberId, onCard }: Opts) {
  const onCardRef = useRef(onCard);
  onCardRef.current = onCard;
  const unmounted = useRef(false);
  const delay = useRef(1000);

  useEffect(() => {
    if (!subscriberId) return;
    unmounted.current = false;
    delay.current = 1000;

    function connect() {
      if (unmounted.current) return;
      const ws = new WebSocket(wsStreamUrl(subscriberId!));

      ws.onopen = () => { delay.current = 1000; };

      ws.onmessage = (e) => {
        try {
          const card = JSON.parse(e.data as string) as ContextCard;
          if (card.text) onCardRef.current(card);
        } catch {
          if (typeof e.data === "string" && e.data.trim()) {
            onCardRef.current({
              chunk_id: crypto.randomUUID(),
              text: e.data,
              ts: Date.now() / 1000,
            });
          }
        }
      };

      ws.onclose = () => {
        if (unmounted.current) return;
        const d = Math.min(delay.current, 30_000);
        delay.current = Math.min(delay.current * 2, 30_000);
        setTimeout(connect, d);
      };

      ws.onerror = () => ws.close();
      return ws;
    }

    const ws = connect();
    return () => {
      unmounted.current = true;
      ws?.close();
    };
  }, [subscriberId]);
}

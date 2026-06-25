/**
 * ApexContext — single SSE connection shared across the whole app.
 * Instantiated once at the App level; all components read from this.
 */
import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export interface SignalState {
  activity_type?: string;
  velocity?: number;
  label?: string;
  confidence?: number;
  urgency?: boolean;
}

export interface PipelineState {
  action?: string;
  label?: string;
  tau?: number;
  reason?: string;
}

export interface MetricsState {
  prp?: number | null;
  ltc?: number | null;
  dps?: number | null;
}

export interface ApexState {
  signal: SignalState;
  pipeline: PipelineState;
  metrics: MetricsState;
  buffer: Record<string, number>;
  context: string;
  connected: boolean;
}

const INITIAL: ApexState = {
  signal: {}, pipeline: {}, metrics: {},
  buffer: {}, context: "", connected: false,
};

const ApexContext = createContext<ApexState>(INITIAL);

export function ApexProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ApexState>(INITIAL);

  useEffect(() => {
    let src: EventSource;
    let timer: ReturnType<typeof setTimeout>;

    function connect() {
      src = new EventSource("/events");
      src.onopen = () => setState((s) => ({ ...s, connected: true }));
      src.onerror = () => {
        setState((s) => ({ ...s, connected: false }));
        src.close();
        timer = setTimeout(connect, 3000);
      };
      (["signal", "pipeline", "metrics", "buffer", "context"] as const).forEach(
        (type) => {
          src.addEventListener(type, (e) => {
            try {
              const data = JSON.parse((e as MessageEvent).data);
              setState((s) => ({ ...s, [type]: data }));
            } catch { /* ignore */ }
          });
        }
      );
    }

    connect();
    return () => { clearTimeout(timer); src?.close(); };
  }, []);

  return <ApexContext.Provider value={state}>{children}</ApexContext.Provider>;
}

export function useApex() {
  return useContext(ApexContext);
}

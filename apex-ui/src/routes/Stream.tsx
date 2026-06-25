/**
 * Stream — main view (requires auth).
 * Sidebar (always dark) + CardFeed (domain-themed main pane).
 * Applies domain CSS tokens from the logged-in user's profile.
 */
import React, { useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Sidebar } from "@/features/sidebar/Sidebar";
import { CardFeed } from "@/features/cards/CardFeed";
import { getTokens, getDomainLabel } from "@/theme/tokens";

export default function Stream() {
  const { user } = useAuth();

  // Apply domain theme tokens to root
  useEffect(() => {
    if (!user) return;
    const tokens = getTokens(user.domain);
    for (const [k, v] of Object.entries(tokens)) {
      document.documentElement.style.setProperty(k, v);
    }
    document.documentElement.setAttribute("data-domain", user.domain);
  }, [user?.domain]);

  if (!user) return null;

  return (
    <div className="flex h-full" style={{ background: "var(--bg)" }}>
      <Sidebar />

      {/* Main pane */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Pane header */}
        <div
          className="flex items-center gap-3 px-5 py-3 border-b shrink-0"
          style={{ borderColor: "var(--border)", background: "var(--surface)" }}
        >
          <h1 className="text-sm font-semibold" style={{ color: "var(--text)" }}>
            {getDomainLabel(user.domain)} context stream
          </h1>
          <span className="text-xs ml-2" style={{ color: "var(--text-muted)" }}>
            {user.username}
          </span>
        </div>

        {/* Feed */}
        <div className="flex-1 overflow-hidden">
          <CardFeed
            userId={user.user_id}
            subscriberId={user.subscriber_id}
            domain={user.domain}
          />
        </div>
      </div>
    </div>
  );
}

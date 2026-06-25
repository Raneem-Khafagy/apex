/**
 * Sidebar — persistent dark left rail (Open WebUI style).
 * Shows user identity, domain badge, APEX status, card history, nav links.
 * Always dark regardless of the current domain theme.
 */
import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useApex } from "@/contexts/ApexContext";
import { getDomainLabel, getDomainIcon } from "@/theme/tokens";
import { getCards } from "@/lib/storage";

const SIDEBAR_BG   = "#111827";
const SIDEBAR_BDR  = "#1f2937";
const SIDEBAR_TEXT = "#f9fafb";
const SIDEBAR_MUTED= "#9ca3af";
const SIDEBAR_ITEM = "#1f2937";

export function Sidebar() {
  const { user, logout } = useAuth();
  const { connected } = useApex();
  const navigate = useNavigate();

  const recentCards = user
    ? getCards(user.user_id).slice(0, 8)
    : [];

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <aside
      className="flex flex-col h-full w-60 shrink-0 overflow-hidden"
      style={{
        background: SIDEBAR_BG,
        borderRight: `1px solid ${SIDEBAR_BDR}`,
        color: SIDEBAR_TEXT,
        fontFamily: "Inter, system-ui, sans-serif",
        fontSize: 13,
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-2 px-4 py-3 border-b"
        style={{ borderColor: SIDEBAR_BDR }}
      >
        <span style={{ color: "#6366f1", fontSize: 18, fontWeight: 700 }}>⬡</span>
        <span className="font-bold tracking-wide" style={{ color: "#e0e7ff" }}>APEX</span>
        <span
          className="ml-auto text-xs"
          style={{ color: connected ? "#22c55e" : "#6b7280" }}
        >
          {connected ? "● Live" : "○ Off"}
        </span>
      </div>

      {/* User identity */}
      {user && (
        <div
          className="px-4 py-3 border-b"
          style={{ borderColor: SIDEBAR_BDR }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div
              className="flex items-center justify-center rounded-full text-xs font-bold"
              style={{
                width: 28, height: 28,
                background: "#6366f1",
                color: "#fff",
                flexShrink: 0,
              }}
            >
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div className="overflow-hidden">
              <p className="font-medium truncate" style={{ color: SIDEBAR_TEXT }}>
                {user.username}
              </p>
            </div>
          </div>
          <span
            className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
            style={{ background: SIDEBAR_ITEM, color: SIDEBAR_MUTED }}
          >
            {getDomainIcon(user.domain)} {getDomainLabel(user.domain)}
          </span>
        </div>
      )}

      {/* Nav */}
      <nav className="px-2 py-2 border-b" style={{ borderColor: SIDEBAR_BDR }}>
        <NavLink
          to="/stream"
          className={({ isActive }) =>
            `flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors ${
              isActive ? "font-semibold" : ""
            }`
          }
          style={({ isActive }) => ({
            background: isActive ? SIDEBAR_ITEM : "transparent",
            color: isActive ? SIDEBAR_TEXT : SIDEBAR_MUTED,
          })}
        >
          ◈ Context stream
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors ${
              isActive ? "font-semibold" : ""
            }`
          }
          style={({ isActive }) => ({
            background: isActive ? SIDEBAR_ITEM : "transparent",
            color: isActive ? SIDEBAR_TEXT : SIDEBAR_MUTED,
          })}
        >
          ⚙ Settings
        </NavLink>
      </nav>

      {/* Recent history */}
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {recentCards.length > 0 && (
          <>
            <p
              className="px-3 py-1 text-xs font-semibold uppercase tracking-widest"
              style={{ color: SIDEBAR_MUTED }}
            >
              Recent
            </p>
            {recentCards.map((card) => (
              <div
                key={card.chunk_id}
                className="px-3 py-1.5 rounded text-xs truncate cursor-default"
                style={{ color: SIDEBAR_MUTED }}
                title={card.text}
              >
                {card.text.slice(0, 50)}
              </div>
            ))}
          </>
        )}
      </div>

      {/* Bottom: logout */}
      <div className="px-2 py-2 border-t" style={{ borderColor: SIDEBAR_BDR }}>
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors hover:bg-gray-800"
          style={{ color: SIDEBAR_MUTED }}
        >
          ⎋ Sign out
        </button>
      </div>
    </aside>
  );
}

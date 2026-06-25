/**
 * App — root component.
 *
 * - AuthProvider (JWT state, /auth/me)
 * - ApexProvider (single SSE /events connection)
 * - Auth guard: unauthenticated → /login, not onboarded → /onboarding
 * - Cmd/Ctrl+Shift+R toggles MetricsOverlay
 */
import React, { lazy, Suspense, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { ApexProvider } from "@/contexts/ApexContext";
import { MetricsOverlay } from "@/features/metrics/MetricsOverlay";
import { Sidebar } from "@/features/sidebar/Sidebar";

// Lazy-load routes — each becomes its own chunk, cutting initial bundle
const Login      = lazy(() => import("@/routes/Login"));
const Onboarding = lazy(() => import("@/routes/Onboarding"));
const Stream     = lazy(() => import("@/routes/Stream"));
const Settings   = lazy(() => import("@/routes/Settings"));

const PageFallback = (
  <div className="flex items-center justify-center min-h-screen"
    style={{ background: "#0d1117", color: "#9ca3af", fontSize: 13 }}>
    Loading…
  </div>
);

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}

function AppShell() {
  const { user, loading } = useAuth();
  const [metrics, setMetrics] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "R") {
        e.preventDefault();
        setMetrics((v) => !v);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen"
        style={{ background: "#0d1117", color: "#9ca3af", fontSize: 13 }}>
        Loading…
      </div>
    );
  }

  return (
    <div className="flex flex-col" style={{ height: "100dvh", overflow: "hidden" }}>
      <div className="flex-1 overflow-hidden">
        <Suspense fallback={PageFallback}>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<Login />} />

          {/* Semi-public: show only when logged in */}
          <Route
            path="/onboarding"
            element={user ? <Onboarding /> : <Navigate to="/login" replace />}
          />

          {/* Protected: require auth + onboarding */}
          <Route
            path="/stream"
            element={
              !user ? (
                <Navigate to="/login" replace />
              ) : !user.onboarded ? (
                <Navigate to="/onboarding" replace />
              ) : (
                <AuthedLayout>
                  <Stream />
                </AuthedLayout>
              )
            }
          />
          <Route
            path="/settings"
            element={
              !user ? (
                <Navigate to="/login" replace />
              ) : !user.onboarded ? (
                <Navigate to="/onboarding" replace />
              ) : (
                <AuthedLayout>
                  <Settings />
                </AuthedLayout>
              )
            }
          />

          {/* Default redirect */}
          <Route
            path="*"
            element={
              <Navigate
                to={!user ? "/login" : !user.onboarded ? "/onboarding" : "/stream"}
                replace
              />
            }
          />
        </Routes>
        </Suspense>
      </div>

      {/* Researcher metrics strip */}
      {metrics && user && <MetricsOverlay />}
    </div>
  );
}

/**
 * Shared layout wrapper for authenticated routes.
 * ApexProvider (SSE connection) lives here — it only mounts when the
 * user is actually authenticated, avoiding wasted connections on the
 * login / onboarding pages.
 */
function AuthedLayout({ children }: { children: React.ReactNode }) {
  return (
    <ApexProvider>
      <div className="flex h-full">
        <Sidebar />
        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
    </ApexProvider>
  );
}

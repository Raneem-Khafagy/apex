/**
 * AuthContext — global authentication state.
 *
 * Holds the decoded user + subscriber_id after a successful /auth/me call.
 * All authenticated routes read from this context.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import type { Domain } from "@/lib/api";
import { getMe, login as apiLogin, register as apiRegister } from "@/lib/api";
import { saveToken, getToken, clearToken } from "@/lib/storage";

export interface AuthUser {
  user_id: string;
  username: string;
  domain: Domain;
  subscriber_id: string;
  onboarded: boolean;
  profile_json: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, domain: Domain) => Promise<void>;
  logout: () => void;
  refreshMe: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  login: async () => {},
  register: async () => {},
  logout: () => {},
  refreshMe: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshMe = useCallback(async () => {
    const token = getToken();
    if (!token) { setLoading(false); return; }

    // Fast path: use sessionStorage cache so hard-refresh doesn't block on /auth/me
    const cached = sessionStorage.getItem("apex.me");
    if (cached) {
      try {
        setUser(JSON.parse(cached) as AuthUser);
        setLoading(false);
      } catch { /* fall through to network */ }
    }

    // Network refresh (with 5 s timeout so a down server doesn't hang forever)
    try {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 5000);
      const me = await getMe();
      clearTimeout(tid);
      sessionStorage.setItem("apex.me", JSON.stringify(me));
      setUser(me as AuthUser);
    } catch {
      sessionStorage.removeItem("apex.me");
      clearToken();
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // On mount, try to restore session from stored token
  useEffect(() => { refreshMe(); }, [refreshMe]);

  const login = useCallback(async (username: string, password: string) => {
    const res = await apiLogin(username, password);
    saveToken(res.token);
    sessionStorage.removeItem("apex.me");
    const me = await getMe();
    sessionStorage.setItem("apex.me", JSON.stringify(me));
    setUser(me as AuthUser);
  }, []);

  const register = useCallback(
    async (username: string, password: string, domain: Domain) => {
      const res = await apiRegister(username, password, domain);
      saveToken(res.token);
      sessionStorage.removeItem("apex.me");
      const me = await getMe();
      sessionStorage.setItem("apex.me", JSON.stringify(me));
      setUser(me as AuthUser);
    },
    []
  );

  const logout = useCallback(() => {
    clearToken();
    sessionStorage.removeItem("apex.me");
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, refreshMe }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

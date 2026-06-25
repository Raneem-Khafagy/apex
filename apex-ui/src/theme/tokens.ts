/**
 * Design tokens for domain themes.
 *
 * The three built-in themes (writing / factory / research) are presets.
 * Any other domain string falls back to DEFAULT_TOKENS — a clean dark theme.
 * Use getTokens(domain) everywhere rather than indexing DOMAIN_TOKENS directly.
 */
import type { ConsumerProfile } from "@/lib/api";

export interface DomainTokens {
  "--bg": string;
  "--surface": string;
  "--surface-2": string;
  "--border": string;
  "--text": string;
  "--text-muted": string;
  "--accent": string;
  "--accent-dim": string;
  "--danger": string;
  "--card-radius": string;
  "--font-body": string;
  "--font-size-body": string;
}

// ── Preset themes ─────────────────────────────────────────────────────────────

export const DOMAIN_TOKENS: Record<string, DomainTokens> = {
  writing: {
    "--bg":             "#F5F2EB",
    "--surface":        "#FDFBF7",
    "--surface-2":      "#EDEAE1",
    "--border":         "#D8D4C8",
    "--text":           "#1C1C1E",
    "--text-muted":     "#6B6860",
    "--accent":         "#4F46E5",
    "--accent-dim":     "#EEF2FF",
    "--danger":         "#DC2626",
    "--card-radius":    "12px",
    "--font-body":      "'Source Serif Pro', Georgia, serif",
    "--font-size-body": "16px",
  },
  factory: {
    "--bg":             "#0A0E1A",
    "--surface":        "#111827",
    "--surface-2":      "#1F2937",
    "--border":         "#374151",
    "--text":           "#F9FAFB",
    "--text-muted":     "#9CA3AF",
    "--accent":         "#F59E0B",
    "--accent-dim":     "#1C160033",
    "--danger":         "#EF4444",
    "--card-radius":    "4px",
    "--font-body":      "'JetBrains Mono', 'Fira Code', monospace",
    "--font-size-body": "14px",
  },
  research: {
    "--bg":             "#FAF8F3",
    "--surface":        "#FFFEF9",
    "--surface-2":      "#F0EDE5",
    "--border":         "#D4D0C8",
    "--text":           "#262626",
    "--text-muted":     "#5C5A54",
    "--accent":         "#0D9488",
    "--accent-dim":     "#ECFDF5",
    "--danger":         "#DC2626",
    "--card-radius":    "8px",
    "--font-body":      "Charter, Georgia, serif",
    "--font-size-body": "16px",
  },
};

/** Fallback for any domain without a preset theme. */
export const DEFAULT_TOKENS: DomainTokens = {
  "--bg":             "#0d1117",
  "--surface":        "#161b22",
  "--surface-2":      "#1f2937",
  "--border":         "#30363d",
  "--text":           "#f0f6fc",
  "--text-muted":     "#8b949e",
  "--accent":         "#6366f1",
  "--accent-dim":     "#1e1b4b44",
  "--danger":         "#f85149",
  "--card-radius":    "8px",
  "--font-body":      "Inter, system-ui, sans-serif",
  "--font-size-body": "14px",
};

/** Returns the preset theme for well-known domains, DEFAULT_TOKENS otherwise. */
export function getTokens(domain: string): DomainTokens {
  return DOMAIN_TOKENS[domain.toLowerCase()] ?? DEFAULT_TOKENS;
}

// ── Domain labels & icons ─────────────────────────────────────────────────────

const PRESET_LABELS: Record<string, string> = {
  writing:  "Writing",
  factory:  "Factory",
  research: "Research",
};

const PRESET_ICONS: Record<string, string> = {
  writing:  "✍",
  factory:  "⚙",
  research: "🔬",
};

/** Capitalises first letter of any domain string, uses preset label when available. */
export function getDomainLabel(domain: string): string {
  return PRESET_LABELS[domain.toLowerCase()]
    ?? (domain.charAt(0).toUpperCase() + domain.slice(1));
}

/** Returns an icon for the domain, or ◈ for unknown domains. */
export function getDomainIcon(domain: string): string {
  return PRESET_ICONS[domain.toLowerCase()] ?? "◈";
}

/**
 * @deprecated Use getTokens / getDomainLabel / getDomainIcon instead.
 * Kept so existing tests that import these names continue to pass.
 */
export const DOMAIN_LABELS: Record<string, string> = PRESET_LABELS;
export const DOMAIN_ICONS: Record<string, string>  = PRESET_ICONS;

// ── Default ConsumerProfiles ──────────────────────────────────────────────────

const PRESET_PROFILES: Record<string, ConsumerProfile> = {
  writing: {
    autonomy_level:     "assistive",
    goal_horizon:       "short",
    interaction_style:  "ambient",
    output_format:      "markdown",
    vocabulary_level:   "domain-expert",
    verbosity:          "concise",
    citation_style:     "inline",
    max_context_tokens: 512,
  },
  factory: {
    autonomy_level:     "autonomous",
    goal_horizon:       "short",
    interaction_style:  "hard-interrupt",
    output_format:      "structured-alert",
    vocabulary_level:   "technical",
    verbosity:          "concise",
    citation_style:     "none",
    max_context_tokens: 256,
  },
  research: {
    autonomy_level:     "suggestive",
    goal_horizon:       "long",
    interaction_style:  "conversational",
    output_format:      "markdown",
    vocabulary_level:   "domain-expert",
    verbosity:          "detailed",
    citation_style:     "footnote",
    max_context_tokens: 1024,
  },
};

/** Sensible generic defaults for any domain without a preset profile. */
export const GENERIC_PROFILE: ConsumerProfile = {
  autonomy_level:     "assistive",
  goal_horizon:       "mid",
  interaction_style:  "soft-interrupt",
  output_format:      "markdown",
  vocabulary_level:   "domain-expert",
  verbosity:          "standard",
  citation_style:     "inline",
  max_context_tokens: 512,
};

/** Returns preset profile for known domains, GENERIC_PROFILE otherwise. */
export function getDefaultProfile(domain: string): ConsumerProfile {
  return PRESET_PROFILES[domain.toLowerCase()] ?? GENERIC_PROFILE;
}

/**
 * @deprecated Use getDefaultProfile(domain) instead.
 * Kept for backward compat with existing tests.
 */
export const DEFAULT_PROFILES: Record<string, ConsumerProfile> = PRESET_PROFILES;

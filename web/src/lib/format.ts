/** Display helpers — percentages as whole numbers; tabular-friendly strings. */

import type { MatchPrediction } from "./types";

export function pct(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return String(Math.round(p * 100));
}

export function pctLabel(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${Math.round(p * 100)}%`;
}

export function emptyStripCopy(kind: "model" | "book"): string {
  return kind === "model" ? "No model line" : "No book line";
}

function finite(p: number | null | undefined): p is number {
  return p != null && !Number.isNaN(p);
}

/** True when de-vigged home / draw / away are all present. */
export function hasBookOneXTwo(match: MatchPrediction): boolean {
  return (
    finite(match.book_p_home) &&
    finite(match.book_p_draw) &&
    finite(match.book_p_away)
  );
}

/**
 * Model BTTS + O2.5, plus book O2.5 when present.
 * Omits book BTTS (no source yet — PRD open Q). No mid-dot separators.
 * A null book source stays "no book line" — it is not labelled "book".
 */
export function secondaryMarketsLine(match: MatchPrediction): string {
  const bits = [
    `BTTS ${pctLabel(match.p_btts_yes)}`,
    `O2.5 ${pctLabel(match.p_over_25)}`,
  ];
  const bookOver = finite(match.book_p_over_25);
  if (bookOver) {
    bits.push(`book O2.5 ${pctLabel(match.book_p_over_25)}`);
  }
  const line = bits.join(" / ");
  if (match.book_source) {
    return `${line} — ${match.book_source}`;
  }
  if (!hasBookOneXTwo(match) && !bookOver) {
    return `${line} — no book line`;
  }
  return line;
}

export function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/London",
  }).format(d);
}

export function resultLabel(result: string | null | undefined): string {
  if (!result) return "—";
  if (result === "H") return "Home";
  if (result === "D") return "Draw";
  if (result === "A") return "Away";
  return result;
}

/**
 * Top 1X2 outcome from model probs.
 * Prefer `topOneXTwo` from `./metrics` for new call sites (tie-break documented there).
 */
export function topOneXTwo(pHome: number, pDraw: number, pAway: number): "H" | "D" | "A" {
  if (pHome >= pDraw && pHome >= pAway) return "H";
  if (pDraw >= pAway) return "D";
  return "A";
}

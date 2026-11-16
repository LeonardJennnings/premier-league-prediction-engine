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

/**
 * Model BTTS + O2.5, plus book O2.5 when present.
 * Omits book BTTS (no source yet — PRD open Q). No mid-dot separators.
 */
export function secondaryMarketsLine(match: MatchPrediction): string {
  const bits = [
    `BTTS ${pctLabel(match.p_btts_yes)}`,
    `O2.5 ${pctLabel(match.p_over_25)}`,
  ];
  if (match.book_p_over_25 != null && !Number.isNaN(match.book_p_over_25)) {
    bits.push(`book O2.5 ${pctLabel(match.book_p_over_25)}`);
  }
  const source = match.book_source ?? "book";
  return `${bits.join(" / ")} — ${source}`;
}

export function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
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

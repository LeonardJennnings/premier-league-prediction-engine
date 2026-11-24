/**
 * Weekend slate selection.
 * Live API (still to kick off) → committed snapshot (still to kick off) →
 * labelled sample. A real snapshot whose kickoffs have all passed stays empty:
 * sample fixtures are not swapped in for a published slate.
 */

import type { MatchPrediction } from "./types";

/** Same London layout as `formatKickoff`. Kept local so this module stays importable from Node. */
function formatKickoffLondon(iso: string): string {
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

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const FINISHED = new Set(["settled", "cancelled", "canceled", "finished"]);

export function kickoffMs(iso: string): number | null {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t;
}

export function stillToKickOff(match: MatchPrediction, nowMs: number): boolean {
  const status = (match.status || "").toLowerCase();
  if (FINISHED.has(status)) return false;
  const t = kickoffMs(match.kickoff_utc);
  if (t == null) return true;
  return t > nowMs;
}

export function futureMatches(
  matches: MatchPrediction[],
  nowMs: number,
): MatchPrediction[] {
  return matches.filter((match) => stillToKickOff(match, nowMs));
}

export function latestKickoffIso(matches: MatchPrediction[]): string | null {
  let best: string | null = null;
  let bestMs = Number.NEGATIVE_INFINITY;
  for (const match of matches) {
    const t = kickoffMs(match.kickoff_utc);
    if (t != null && t > bestMs) {
      bestMs = t;
      best = match.kickoff_utc;
    }
  }
  return best;
}

/** Roll a labelled sample forward by whole weeks so the preview stays upcoming. */
export function shiftSampleKickoffs(
  matches: MatchPrediction[],
  nowMs: number,
): MatchPrediction[] {
  const times = matches
    .map((match) => kickoffMs(match.kickoff_utc))
    .filter((t): t is number => t != null);
  if (times.length === 0) return matches;
  const earliest = Math.min(...times);
  const latest = Math.max(...times);
  if (latest > nowMs) return matches;
  let shift = 0;
  while (earliest + shift <= nowMs) shift += WEEK_MS;
  return matches.map((match) => {
    const t = kickoffMs(match.kickoff_utc);
    if (t == null) return match;
    return { ...match, kickoff_utc: toIsoZ(t + shift) };
  });
}

function toIsoZ(ms: number): string {
  return new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
}

export type SlateSource = "live" | "snapshot" | "sample" | "empty";

export type SlateChoice = {
  source: SlateSource;
  matches: MatchPrediction[];
  modelVersion: string;
  notice: string | null;
};

export type SlateInput = {
  nowMs: number;
  apiBase: string;
  live:
    | { ok: true; modelVersion: string; matches: MatchPrediction[] }
    | { ok: false; status: number; detail: string };
  snapshot: { modelVersion: string; matches: MatchPrediction[] } | null;
  sampleMatches: MatchPrediction[];
};

function apiClause(input: SlateInput): string {
  if (input.live.ok) return "Live API returned no upcoming rows.";
  if (input.live.status === 0) return `API unreachable at ${input.apiBase}.`;
  return `API ${input.live.status}: ${input.live.detail}.`;
}

export function chooseWeekendSlate(input: SlateInput): SlateChoice {
  const liveFuture = input.live.ok
    ? futureMatches(input.live.matches, input.nowMs)
    : [];
  if (liveFuture.length > 0 && input.live.ok) {
    return {
      source: "live",
      matches: liveFuture,
      modelVersion: input.live.modelVersion,
      notice: null,
    };
  }

  const snapshot = input.snapshot;
  const snapFuture = snapshot
    ? futureMatches(snapshot.matches, input.nowMs)
    : [];
  if (snapshot && snapFuture.length > 0) {
    const dropped = snapshot.matches.length - snapFuture.length;
    const hidden =
      dropped > 0
        ? ` ${dropped} kicked-off snapshot ${dropped === 1 ? "match stays" : "matches stay"} off this page.`
        : "";
    return {
      source: "snapshot",
      matches: snapFuture,
      modelVersion: snapshot.modelVersion,
      notice: `${apiClause(input)} Showing matches still to kick off from the committed publish snapshot.${hidden}`,
    };
  }

  if (snapshot) {
    const last = latestKickoffIso(snapshot.matches);
    const when = last ? formatKickoffLondon(last) : "unknown";
    return {
      source: "empty",
      matches: [],
      modelVersion: snapshot.modelVersion,
      notice: `${apiClause(input)} The committed snapshot's last kickoff was ${when}. Nothing in that slate is still to kick off. The past scorecard updates after settlement.`,
    };
  }

  const shifted = shiftSampleKickoffs(input.sampleMatches, input.nowMs);
  const sampleFuture = futureMatches(shifted, input.nowMs);
  const rolled = shifted.some(
    (match, index) => match.kickoff_utc !== input.sampleMatches[index]?.kickoff_utc,
  );
  if (sampleFuture.length > 0) {
    const rollNote = rolled
      ? " Sample kickoffs are rolled forward by whole weeks so this labelled preview stays upcoming."
      : "";
    const lead = input.live.ok
      ? "No upcoming matches published yet. Showing sample slate."
      : input.live.status === 0
        ? `API unreachable at ${input.apiBase}. Showing sample slate.`
        : `API ${input.live.status}: ${input.live.detail}. Showing sample slate.`;
    return {
      source: "sample",
      matches: sampleFuture,
      modelVersion: "sample-local",
      notice: `${lead}${rollNote}`,
    };
  }

  return {
    source: "empty",
    matches: [],
    modelVersion: "sample-local",
    notice: `${apiClause(input)} No sample matches are still to kick off.`,
  };
}

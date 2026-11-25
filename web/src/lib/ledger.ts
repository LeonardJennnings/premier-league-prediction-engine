/**
 * Read the committed ledger site files. The public page does not need an API.
 * Sample rows are allowed only when PUBLIC_SHOW_SAMPLE is exactly "1".
 */

import { pctLabel } from "./format.ts";
import { topOneXTwo, type OutcomeCode } from "./metrics.ts";
import type { MatchPrediction } from "./types.ts";

export type LedgerScore = {
  home: number;
  away: number;
  probability: number;
};

export type LedgerBook = {
  kind: string;
  captured_at: string | null;
  source: string | null;
  p_home: number | null;
  p_draw: number | null;
  p_away: number | null;
  p_over_25: number | null;
  p_under_25: number | null;
};

export type LedgerMatch = {
  match_id: number;
  matchday: number | null;
  kickoff_utc: string;
  home_team: string;
  away_team: string;
  model_version: string;
  status: string;
  home_goals: number | null;
  away_goals: number | null;
  result: OutcomeCode | null;
  most_likely_score: LedgerScore | null;
  p_home: number;
  p_draw: number;
  p_away: number;
  p_btts_yes: number;
  p_btts_no: number;
  p_over_25: number;
  p_under_25: number;
  lambda_home: number | null;
  lambda_away: number | null;
  score_omitted_reason: string | null;
  book: LedgerBook | null;
};

export type MatchdayGroup<T> = {
  matchday: number | null;
  matches: T[];
};

export type LedgerFile = {
  note?: string;
  matchdays: MatchdayGroup<LedgerMatch>[];
};

export type ScorecardFile = {
  note: string;
  counts: {
    settled: number;
    pending: number;
    postponed: number;
    void: number;
  };
  home_draw_away: { match_count: number; model_log_loss: number | null };
  book_comparison: {
    match_count: number;
    model_log_loss: number | null;
    closing_book_log_loss: number | null;
    gap_model_minus_closing: number | null;
    range_95: number[] | null;
  };
  exact_score: {
    match_count: number;
    hits: number;
    always_1_1_hits: number;
    mean_printed_probability: number | null;
  };
  goal_error: { match_count: number; mean_absolute_error: number | null };
};

export type ExactCall = "hit" | "miss" | "not stored";

export type MatchGrade = {
  scoreText: string | null;
  scorePct: string | null;
  resultName: string;
  resultPct: string;
  expectedGoals: string | null;
  disagree: boolean;
  omittedReason: string | null;
  actualText: string | null;
  exact: ExactCall | null;
  resultCall: "hit" | "miss" | null;
};

const RESULT_NAME: Record<OutcomeCode, string> = {
  H: "Home",
  D: "Draw",
  A: "Away",
};

export function sampleAllowed(flag: string | undefined | null): boolean {
  return flag === "1";
}

export function kickoffMs(iso: string): number | null {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t;
}

/** Upcoming rows whose kickoff is still ahead. A played match never stays here. */
export function visibleUpcoming(file: LedgerFile, nowMs: number): MatchdayGroup<LedgerMatch>[] {
  return file.matchdays
    .map((day) => ({
      matchday: day.matchday,
      matches: day.matches.filter((match) => stillUpcoming(match, nowMs)),
    }))
    .filter((day) => day.matches.length > 0);
}

export function flattenMatches(days: MatchdayGroup<LedgerMatch>[]): LedgerMatch[] {
  return days.flatMap((day) => day.matches);
}

export function gradeMatch(match: LedgerMatch): MatchGrade {
  const pick = topOneXTwo({
    home: match.p_home,
    draw: match.p_draw,
    away: match.p_away,
  });
  const pickP =
    pick === "H" ? match.p_home : pick === "D" ? match.p_draw : match.p_away;
  const score = match.most_likely_score;
  const scoreCode = score ? scoreCodeOf(score.home, score.away) : null;
  const played = match.status === "settled" && match.home_goals != null && match.away_goals != null;
  let exact: ExactCall | null = null;
  if (played) {
    if (!score) exact = "not stored";
    else if (score.home === match.home_goals && score.away === match.away_goals) exact = "hit";
    else exact = "miss";
  }
  let resultCall: "hit" | "miss" | null = null;
  if (played && match.result) {
    resultCall = pick === match.result ? "hit" : "miss";
  }
  return {
    scoreText: score ? `${score.home}-${score.away}` : null,
    scorePct: score ? pctLabel(score.probability) : null,
    resultName: RESULT_NAME[pick],
    resultPct: pctLabel(pickP),
    expectedGoals:
      match.lambda_home != null && match.lambda_away != null
        ? `${match.lambda_home.toFixed(2)} home, ${match.lambda_away.toFixed(2)} away`
        : null,
    disagree: scoreCode != null && scoreCode !== pick,
    omittedReason: score ? null : match.score_omitted_reason,
    actualText: played ? `${match.home_goals}-${match.away_goals}` : null,
    exact,
    resultCall,
  };
}

/** Exact-score tile. Postponed, pending, and rows with no stored score stay out. */
export function exactScoreSummary(matches: LedgerMatch[]) {
  const rows = matches.filter(
    (match) =>
      match.status === "settled" &&
      match.home_goals != null &&
      match.away_goals != null &&
      match.most_likely_score != null,
  );
  let hits = 0;
  let always = 0;
  let printed = 0;
  for (const match of rows) {
    const score = match.most_likely_score!;
    if (score.home === match.home_goals && score.away === match.away_goals) hits += 1;
    if (match.home_goals === 1 && match.away_goals === 1) always += 1;
    printed += score.probability;
  }
  return {
    match_count: rows.length,
    hits,
    always_1_1_hits: always,
    mean_printed_probability: rows.length ? printed / rows.length : null,
  };
}

/** Correct-result hits on every settled match, with the average probability of that top pick. */
export function resultHitSummary(matches: LedgerMatch[]) {
  const rows = matches.filter(
    (match) => match.status === "settled" && match.result != null && match.home_goals != null,
  );
  let hits = 0;
  let printed = 0;
  for (const match of rows) {
    const pick = topOneXTwo({
      home: match.p_home,
      draw: match.p_draw,
      away: match.p_away,
    });
    const p = pick === "H" ? match.p_home : pick === "D" ? match.p_draw : match.p_away;
    printed += p;
    if (pick === match.result) hits += 1;
  }
  return {
    match_count: rows.length,
    hits,
    mean_top_probability: rows.length ? printed / rows.length : null,
  };
}

export function asPrediction(match: LedgerMatch): MatchPrediction {
  const book = match.book;
  return {
    match_id: match.match_id,
    home_team: match.home_team,
    away_team: match.away_team,
    kickoff_utc: match.kickoff_utc,
    matchday: match.matchday,
    competition: "PL",
    status: match.status,
    p_home: match.p_home,
    p_draw: match.p_draw,
    p_away: match.p_away,
    p_btts_yes: match.p_btts_yes,
    p_btts_no: match.p_btts_no,
    p_over_25: match.p_over_25,
    p_under_25: match.p_under_25,
    book_p_home: book?.p_home ?? null,
    book_p_draw: book?.p_draw ?? null,
    book_p_away: book?.p_away ?? null,
    book_p_over_25: book?.p_over_25 ?? null,
    book_p_under_25: book?.p_under_25 ?? null,
    book_source: book?.source ?? null,
    published_at: "",
    as_of_date: "",
    model_version: match.model_version,
    home_goals: match.home_goals,
    away_goals: match.away_goals,
    result: match.result,
    settled_at: null,
  };
}

/** Over/under 2.5 log loss. The book figure uses only matches that have a closing price. */
export function overUnderSummary(matches: LedgerMatch[]) {
  const rows = settledWithScore(matches);
  const model = mean(rows.map((match) => binaryLoss(match.p_over_25, match.p_under_25, over25(match))));
  const priced = rows.filter((match) => match.book?.p_over_25 != null);
  const book = mean(
    priced.map((match) =>
      binaryLoss(match.book!.p_over_25, match.book!.p_under_25, over25(match)),
    ),
  );
  return {
    match_count: rows.length,
    model_log_loss: model,
    closing_book_match_count: priced.length,
    closing_book_log_loss: book,
  };
}

/** Both-teams-to-score log loss for the model. The book does not price this market here. */
export function bothTeamsSummary(matches: LedgerMatch[]) {
  const rows = settledWithScore(matches);
  return {
    match_count: rows.length,
    model_log_loss: mean(
      rows.map((match) => binaryLoss(match.p_btts_yes, match.p_btts_no, bothScored(match))),
    ),
  };
}

/**
 * Cumulative model-minus-closing gap after each settled matchday, in kickoff order.
 * The line is never restarted.
 */
export function runningBookGap(matches: LedgerMatch[]) {
  const rows = settledWithScore(matches)
    .filter((match) => match.book?.kind === "closing" && match.result)
    .slice()
    .sort((a, b) => a.kickoff_utc.localeCompare(b.kickoff_utc) || a.match_id - b.match_id);
  const byDay = new Map<number | null, LedgerMatch[]>();
  for (const match of rows) {
    const bucket = byDay.get(match.matchday) ?? [];
    bucket.push(match);
    byDay.set(match.matchday, bucket);
  }
  let seen = 0;
  let gapSum = 0;
  const points: { matchday: number | null; match_count: number; gap: number }[] = [];
  for (const matchday of [...byDay.keys()]) {
    for (const match of byDay.get(matchday) ?? []) {
      const modelP = resultProbability(match);
      const bookP = bookResultProbability(match);
      if (modelP == null || bookP == null) continue;
      gapSum += naturalLoss(modelP) - naturalLoss(bookP);
      seen += 1;
    }
    if (seen > 0) {
      points.push({ matchday, match_count: seen, gap: gapSum / seen });
    }
  }
  return points;
}

export function formatNat(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "not available";
  return value.toFixed(3);
}

function settledWithScore(matches: LedgerMatch[]): LedgerMatch[] {
  return matches.filter(
    (match) => match.status === "settled" && match.home_goals != null && match.away_goals != null,
  );
}

function over25(match: LedgerMatch): boolean {
  return match.home_goals! + match.away_goals! >= 3;
}

function bothScored(match: LedgerMatch): boolean {
  return match.home_goals! > 0 && match.away_goals! > 0;
}

function binaryLoss(yes: number | null, no: number | null, happened: boolean): number {
  const p = happened ? yes : no != null ? no : yes == null ? null : 1 - yes;
  return naturalLoss(p ?? 0);
}

function resultProbability(match: LedgerMatch): number | null {
  if (match.result === "H") return match.p_home;
  if (match.result === "D") return match.p_draw;
  if (match.result === "A") return match.p_away;
  return null;
}

function bookResultProbability(match: LedgerMatch): number | null {
  const book = match.book;
  if (!book || !match.result) return null;
  if (match.result === "H") return book.p_home;
  if (match.result === "D") return book.p_draw;
  if (match.result === "A") return book.p_away;
  return null;
}

function naturalLoss(probability: number): number {
  const clipped = Math.min(Math.max(probability, 1e-15), 1);
  return -Math.log(clipped);
}

function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stillUpcoming(match: LedgerMatch, nowMs: number): boolean {
  const status = match.status.toLowerCase();
  if (status === "postponed" || status === "cancelled" || status === "settled" || status === "pending") {
    return false;
  }
  const t = kickoffMs(match.kickoff_utc);
  return t != null && t > nowMs;
}

function scoreCodeOf(home: number, away: number): OutcomeCode {
  if (home > away) return "H";
  if (home < away) return "A";
  return "D";
}

/**
 * Derived metrics for Epic 5 peeks.
 * Mirrors Python eval maths in `src/evaluate.py` / `src/eval/calibration.py`
 * where noted. Documented in Context `docs/epic5-dataviz-notes.md`.
 */

import type { MatchPrediction } from "./types";

export type OutcomeCode = "H" | "D" | "A";

export type ProbaTriple = {
  home: number;
  draw: number;
  away: number;
};

const EPS = 1e-15;

export function modelProbas(m: MatchPrediction): ProbaTriple {
  return { home: m.p_home, draw: m.p_draw, away: m.p_away };
}

/**
 * Top 1X2 call: argmax of (home, draw, away).
 * Ties break left-to-right H → D → A (stable).
 */
export function topOneXTwo(p: ProbaTriple): OutcomeCode {
  let best: OutcomeCode = "H";
  let bestVal = p.home;
  if (p.draw > bestVal) {
    best = "D";
    bestVal = p.draw;
  }
  if (p.away > bestVal) {
    best = "A";
  }
  return best;
}

/** Hit = model's top 1X2 matched settled result. */
export function isHit(m: MatchPrediction): boolean | null {
  if (m.status !== "settled" || m.result == null) return null;
  const code = m.result as OutcomeCode;
  if (code !== "H" && code !== "D" && code !== "A") return null;
  return topOneXTwo(modelProbas(m)) === code;
}

/**
 * Renormalise rounded percent widths so a strip totals 100.
 * Adjusts the largest segment when rounding drifts.
 */
export function pctWidths(p: ProbaTriple): {
  home: number;
  draw: number;
  away: number;
} {
  const raw = [
    Math.round(p.home * 100),
    Math.round(p.draw * 100),
    Math.round(p.away * 100),
  ];
  const sum = raw[0] + raw[1] + raw[2];
  if (sum === 100) return { home: raw[0], draw: raw[1], away: raw[2] };
  if (sum === 0) return { home: 34, draw: 33, away: 33 };
  const adjusted = [...raw];
  const maxIdx = adjusted.indexOf(Math.max(...adjusted));
  adjusted[maxIdx] += 100 - sum;
  return { home: adjusted[0], draw: adjusted[1], away: adjusted[2] };
}

function oneHot(result: OutcomeCode): [number, number, number] {
  if (result === "H") return [1, 0, 0];
  if (result === "D") return [0, 1, 0];
  return [0, 0, 1];
}

function clip(p: number): number {
  return Math.min(1, Math.max(EPS, p));
}

/**
 * Multiclass Brier for one match — mean squared error / 3,
 * matching `src/evaluate.py` → `brier_score`.
 */
export function matchBrier(p: ProbaTriple, result: OutcomeCode): number {
  const y = oneHot(result);
  const pred = [p.home, p.draw, p.away];
  let sum = 0;
  for (let i = 0; i < 3; i++) {
    const d = pred[i] - y[i];
    sum += d * d;
  }
  return sum / 3;
}

/** Mean NLL of the true outcome — matches `log_loss` in evaluate.py. */
export function matchLogLoss(p: ProbaTriple, result: OutcomeCode): number {
  const idx = result === "H" ? 0 : result === "D" ? 1 : 2;
  return -Math.log(clip([p.home, p.draw, p.away][idx]));
}

export type ProperScores = {
  n: number;
  meanBrier: number | null;
  meanLogLoss: number | null;
};

export function properScores(matches: MatchPrediction[]): ProperScores {
  let n = 0;
  let brierSum = 0;
  let llSum = 0;
  for (const m of matches) {
    if (m.status !== "settled" || m.result == null) continue;
    const code = m.result as OutcomeCode;
    if (code !== "H" && code !== "D" && code !== "A") continue;
    const p = modelProbas(m);
    brierSum += matchBrier(p, code);
    llSum += matchLogLoss(p, code);
    n += 1;
  }
  if (n === 0) return { n: 0, meanBrier: null, meanLogLoss: null };
  return { n, meanBrier: brierSum / n, meanLogLoss: llSum / n };
}

export type WaffleCell = {
  hit: boolean;
  title: string;
};

export type WaffleSummary = {
  cells: WaffleCell[];
  hits: number;
  misses: number;
  total: number;
};

/**
 * One cell per settled match. Order: kickoff asc, then match_id.
 * Not a streak heatmap — ledger order.
 */
export function buildWaffle(matches: MatchPrediction[]): WaffleSummary {
  const settled = matches
    .filter((m) => m.status === "settled" && m.result != null)
    .slice()
    .sort((a, b) => {
      const k = a.kickoff_utc.localeCompare(b.kickoff_utc);
      return k !== 0 ? k : a.match_id - b.match_id;
    });

  const cells: WaffleCell[] = settled.map((m) => {
    const hit = isHit(m) === true;
    const top = topOneXTwo(modelProbas(m));
    return {
      hit,
      title: `${m.home_team} vs ${m.away_team}: ${hit ? "Hit" : "Miss"} (top ${top}, result ${m.result})`,
    };
  });

  const hits = cells.filter((c) => c.hit).length;
  return {
    cells,
    hits,
    misses: cells.length - hits,
    total: cells.length,
  };
}

export type BucketRow = {
  bucket: string;
  predictedShare: string;
  observedShare: string;
  n: string;
};

export type CalibrationPoint = {
  predicted: number;
  observed: number;
  n: number;
};

function fmtPct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/**
 * Home-win reliability buckets (viz #6).
 * When model P(home) fell in a band, how often did home win?
 * Distinct from the hit/miss waffle.
 */
export function homeOutcomeBuckets(
  matches: MatchPrediction[],
  nBins = 10,
): { rows: BucketRow[]; points: CalibrationPoint[]; n: number } {
  const edges: number[] = [];
  for (let i = 0; i <= nBins; i++) edges.push(i / nBins);

  const bins: { preds: number[]; outs: number[] }[] = Array.from(
    { length: nBins },
    () => ({ preds: [], outs: [] }),
  );

  let n = 0;
  for (const m of matches) {
    if (m.status !== "settled" || m.result == null) continue;
    const pHome = m.p_home;
    const y = m.result === "H" ? 1 : 0;
    let idx = nBins - 1;
    for (let b = 0; b < nBins; b++) {
      if (pHome >= edges[b] && pHome < edges[b + 1]) {
        idx = b;
        break;
      }
    }
    if (pHome >= 1) idx = nBins - 1;
    bins[idx].preds.push(pHome);
    bins[idx].outs.push(y);
    n += 1;
  }

  const rows: BucketRow[] = [];
  const points: CalibrationPoint[] = [];
  for (let i = 0; i < nBins; i++) {
    const count = bins[i].preds.length;
    if (count === 0) continue;
    const meanPred =
      bins[i].preds.reduce((a, v) => a + v, 0) / count;
    const meanObs = bins[i].outs.reduce((a, v) => a + v, 0) / count;
    const left = Math.round(edges[i] * 100);
    const right = Math.round(edges[i + 1] * 100);
    rows.push({
      bucket: `${left}–${right}%`,
      predictedShare: fmtPct(meanPred),
      observedShare: fmtPct(meanObs),
      n: String(count),
    });
    points.push({ predicted: meanPred, observed: meanObs, n: count });
  }

  return { rows, points, n };
}

export function formatBrier(v: number | null): string {
  if (v == null) return "—";
  return v.toFixed(3);
}

export function formatLogLoss(v: number | null): string {
  if (v == null) return "—";
  return v.toFixed(3);
}

/**
 * Lorem / sample fixtures for empty-API local preview.
 * Not production data — Designer/DS should not treat these as eval truth.
 */

import type { MatchPrediction } from "./types";

const base = {
  competition: "PL",
  matchday: 12,
  p_btts_no: 0.42,
  p_under_25: 0.46,
  book_p_over_25: 0.54,
  book_p_under_25: 0.46,
  book_source: "sample close",
  published_at: "2026-11-14T12:00:00Z",
  as_of_date: "2026-11-14",
  model_version: "sample-local",
  home_goals: null,
  away_goals: null,
  result: null,
  settled_at: null,
  status: "scheduled",
} as const;

export const SAMPLE_UPCOMING: MatchPrediction[] = [
  {
    ...base,
    match_id: 1,
    home_team: "Arsenal",
    away_team: "Chelsea",
    kickoff_utc: "2026-11-21T12:30:00Z",
    p_home: 0.48,
    p_draw: 0.26,
    p_away: 0.26,
    p_btts_yes: 0.58,
    p_over_25: 0.54,
    book_p_home: 0.45,
    book_p_draw: 0.28,
    book_p_away: 0.27,
  },
  {
    ...base,
    match_id: 2,
    home_team: "Liverpool",
    away_team: "Manchester City",
    kickoff_utc: "2026-11-21T17:30:00Z",
    p_home: 0.36,
    p_draw: 0.27,
    p_away: 0.37,
    p_btts_yes: 0.62,
    p_over_25: 0.61,
    book_p_home: 0.34,
    book_p_draw: 0.28,
    book_p_away: 0.38,
  },
  {
    ...base,
    match_id: 3,
    home_team: "Tottenham",
    away_team: "Brighton",
    kickoff_utc: "2026-11-22T14:00:00Z",
    p_home: 0.41,
    p_draw: 0.29,
    p_away: 0.3,
    p_btts_yes: 0.55,
    p_over_25: 0.52,
    book_p_home: 0.39,
    book_p_draw: 0.3,
    book_p_away: 0.31,
  },
  {
    ...base,
    match_id: 4,
    home_team: "Newcastle",
    away_team: "Aston Villa",
    kickoff_utc: "2026-11-22T16:30:00Z",
    p_home: 0.44,
    p_draw: 0.28,
    p_away: 0.28,
    p_btts_yes: 0.57,
    p_over_25: 0.56,
    book_p_home: 0.42,
    book_p_draw: 0.29,
    book_p_away: 0.29,
  },
];

/** Sample hit/miss pattern (~61/100) for waffle shell until DS binds settled weeks. */
export const SAMPLE_WAFFLE: boolean[] = (() => {
  const hits = [
    1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
  ];
  return hits.map((v) => v === 1);
})();

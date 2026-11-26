/** Shapes mirrored from Epic 4 `src/api/schemas.py` / OpenAPI. */

export type MatchPrediction = {
  match_id: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  matchday: number | null;
  competition: string;
  status: string;
  p_home: number;
  p_draw: number;
  p_away: number;
  p_btts_yes: number;
  p_btts_no: number;
  p_over_25: number;
  p_under_25: number;
  book_p_home: number | null;
  book_p_draw: number | null;
  book_p_away: number | null;
  book_p_over_25: number | null;
  book_p_under_25: number | null;
  book_source: string | null;
  published_at: string;
  as_of_date: string;
  model_version: string;
  model_plus_market?: {
    label: string;
    blend_weight: number | null;
    p_home: number;
    p_draw: number;
    p_away: number;
  } | null;
  book_reconstructed_score?: {
    home: number;
    away: number;
    probability: number;
    label: string;
  } | null;
  home_goals: number | null;
  away_goals: number | null;
  result: string | null;
  settled_at: string | null;
};

export type UpcomingResponse = {
  model_version: string;
  matches: MatchPrediction[];
};

export type WeekSummary = {
  iso_week: string;
  match_count: number;
};

export type WeeksResponse = {
  model_version: string;
  weeks: WeekSummary[];
};

export type WeekResponse = {
  model_version: string;
  iso_week: string;
  matches: MatchPrediction[];
};

export type ModelMeta = {
  model_version: string;
  as_of_date: string;
  created_at: string;
  artefact_path: string;
  hyperparams: Record<string, unknown> | null;
};

export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; detail: string };

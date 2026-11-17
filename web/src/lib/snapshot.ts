/**
 * Build-time upcoming snapshot committed after a real publish.
 * Prefer live API → this file → labelled SAMPLE_UPCOMING.
 * Regenerate: `python scripts/export_upcoming_snapshot.py` (repo root).
 */

import type { UpcomingResponse } from "./types";
import raw from "../data/upcoming-snapshot.json";

export const UPCOMING_SNAPSHOT = raw as UpcomingResponse;

export function hasUpcomingSnapshot(): boolean {
  return Array.isArray(UPCOMING_SNAPSHOT.matches) && UPCOMING_SNAPSHOT.matches.length > 0;
}

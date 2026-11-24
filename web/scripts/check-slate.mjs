/**
 * Sanity checks for weekend slate selection and the null-book label.
 * Run via npm run check:metrics.
 */

import { emptyStripCopy, formatKickoff, secondaryMarketsLine } from "../src/lib/format.ts";
import { chooseWeekendSlate } from "../src/lib/slate.ts";

let failed = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    failed += 1;
  } else {
    console.log("ok:", msg);
  }
}

function match(overrides) {
  return {
    match_id: 1,
    home_team: "Man City",
    away_team: "Fulham",
    kickoff_utc: "2026-11-21T12:30:00Z",
    matchday: 11,
    competition: "PL",
    status: "scheduled",
    p_home: 0.6,
    p_draw: 0.24,
    p_away: 0.16,
    p_btts_yes: 0.47,
    p_btts_no: 0.53,
    p_over_25: 0.52,
    p_under_25: 0.48,
    book_p_home: null,
    book_p_draw: null,
    book_p_away: null,
    book_p_over_25: null,
    book_p_under_25: null,
    book_source: null,
    published_at: "2026-11-17T09:25:50Z",
    as_of_date: "2026-11-08",
    model_version: "dc-test",
    home_goals: null,
    away_goals: null,
    result: null,
    settled_at: null,
    ...overrides,
  };
}

const now = Date.parse("2026-11-24T12:00:00Z");
const stale = [
  match({ match_id: 1, kickoff_utc: "2026-11-21T12:30:00Z" }),
  match({ match_id: 2, home_team: "Brentford", away_team: "Everton", kickoff_utc: "2026-11-23T20:00:00Z" }),
];
const sample = [
  match({
    match_id: 9,
    home_team: "Arsenal",
    away_team: "Chelsea",
    model_version: "sample-local",
    book_source: "sample close",
    book_p_home: 0.45,
    book_p_draw: 0.28,
    book_p_away: 0.27,
    kickoff_utc: "2026-11-21T12:30:00Z",
  }),
];

const kickedOff = chooseWeekendSlate({
  nowMs: now,
  apiBase: "http://127.0.0.1:8000",
  live: { ok: false, status: 0, detail: "offline" },
  snapshot: { modelVersion: "dc-test", matches: stale },
  sampleMatches: sample,
});
assert(kickedOff.source === "empty", "stale real snapshot does not become sample");
assert(kickedOff.matches.length === 0, "stale snapshot shows no current matches");
assert(kickedOff.modelVersion === "dc-test", "empty slate keeps the snapshot model version");
assert(
  kickedOff.notice.includes(formatKickoff("2026-11-23T20:00:00Z")),
  "stale notice names the last kickoff in the same London format as the fixture line",
);
assert(!kickedOff.notice.includes("Arsenal"), "stale notice does not mention sample teams");

const mixed = chooseWeekendSlate({
  nowMs: now,
  apiBase: "http://127.0.0.1:8000",
  live: { ok: true, modelVersion: "dc-live", matches: [] },
  snapshot: {
    modelVersion: "dc-test",
    matches: [
      ...stale,
      match({ match_id: 3, kickoff_utc: "2026-11-28T15:00:00Z", home_team: "Leeds", away_team: "Chelsea" }),
    ],
  },
  sampleMatches: sample,
});
assert(mixed.source === "snapshot", "future snapshot rows still win over sample");
assert(mixed.matches.length === 1 && mixed.matches[0].home_team === "Leeds", "only the future snapshot row remains");
assert(mixed.notice.includes("2 kicked-off snapshot matches stay off"), "dropped kickoffs are counted");

const live = chooseWeekendSlate({
  nowMs: now,
  apiBase: "http://127.0.0.1:8000",
  live: {
    ok: true,
    modelVersion: "dc-live",
    matches: [
      match({ match_id: 4, kickoff_utc: "2026-11-21T12:30:00Z" }),
      match({ match_id: 5, kickoff_utc: "2026-11-28T17:30:00Z", home_team: "Wolves" }),
    ],
  },
  snapshot: { modelVersion: "dc-test", matches: stale },
  sampleMatches: sample,
});
assert(live.source === "live" && live.matches.length === 1 && live.matches[0].home_team === "Wolves", "live rows are filtered to kickoffs still ahead");
assert(live.notice === null, "a live future slate has no fallback notice");

const preview = chooseWeekendSlate({
  nowMs: now,
  apiBase: "http://127.0.0.1:8000",
  live: { ok: false, status: 0, detail: "offline" },
  snapshot: null,
  sampleMatches: sample,
});
assert(preview.source === "sample", "no snapshot falls through to labelled sample");
assert(Date.parse(preview.matches[0].kickoff_utc) > now, "sample kickoff is rolled forward");
assert(
  Date.parse(preview.matches[0].kickoff_utc) - Date.parse("2026-11-21T12:30:00Z") === 7 * 24 * 60 * 60 * 1000,
  "sample rolls by a whole week",
);
assert(preview.notice.includes("rolled forward"), "sample notice says the dates moved");
assert(preview.modelVersion === "sample-local", "sample stays labelled sample-local");

const nullBook = secondaryMarketsLine(stale[0]);
assert(nullBook.endsWith("no book line"), "null book source says no book line");
assert(!nullBook.endsWith("— book"), "null book source is not labelled book");
const withSource = secondaryMarketsLine(sample[0]);
assert(withSource.includes("sample close"), "a real book source is still named");
assert(emptyStripCopy("model") === "No model line", "missing model strip is not a book line");
assert(emptyStripCopy("book") === "No book line", "missing book strip stays no book line");

const kick = formatKickoff("2026-11-21T12:30:00Z");
assert(kick.includes("21") && kick.includes("Nov") && kick.includes("12:30"), `kickoff includes the date (${kick})`);
assert(!kick.includes("·"), "kickoff has no mid-dot");

if (failed) {
  process.exit(1);
}
console.log("All slate checks passed.");

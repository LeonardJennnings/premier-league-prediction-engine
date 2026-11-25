/**
 * Acceptance checks for the ledger-backed site.
 * Run via npm run check:metrics, after the static build when dist exists.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  bothTeamsSummary,
  exactScoreSummary,
  gradeMatch,
  overUnderSummary,
  resultHitSummary,
  runningBookGap,
  sampleAllowed,
  visibleUpcoming,
} from "../src/lib/ledger.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

let failed = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    failed += 1;
  } else {
    console.log("ok:", msg);
  }
}

const upcoming = JSON.parse(
  fs.readFileSync(path.join(root, "src/data/ledger-upcoming.json"), "utf8"),
);
const settled = JSON.parse(
  fs.readFileSync(path.join(root, "src/data/ledger-settled.json"), "utf8"),
);
const scorecard = JSON.parse(
  fs.readFileSync(path.join(root, "src/data/ledger-scorecard.json"), "utf8"),
);

const now = Date.parse("2026-11-25T12:00:00Z");
const days = visibleUpcoming(upcoming, now);
const upcomingMatches = days.flatMap((day) => day.matches);
assert(days.some((day) => day.matchday === 12), "matchday 12 is one group");
assert(
  days.find((day) => day.matchday === 12).matches.length === 10,
  "matchday 12 keeps its Monday match with the round",
);
assert(upcomingMatches.every((match) => Date.parse(match.kickoff_utc) > now), "no past kickoff is upcoming");
assert(
  upcomingMatches.every((match) => match.model_version !== "sample-local"),
  "upcoming ledger rows are not sample",
);
const forest = upcomingMatches.find((match) => match.match_id === 560658);
assert(forest.home_team === "Nott'm Forest" && forest.away_team === "Chelsea", "Forest v Chelsea is listed");
const forestGrade = gradeMatch(forest);
assert(forestGrade.scoreText === "1-1", "Forest predicted score is 1-1");
assert(forestGrade.disagree, "Forest favourite score and favourite result disagree");
assert(forestGrade.expectedGoals != null, "Forest expected goals are shown");

const settledMatches = settled.matchdays.flatMap((day) => day.matches);
assert(settledMatches.length === 10, "every settled matchday 11 row is listed");
assert(
  settledMatches.every((match) => gradeMatch(match).exact === "not stored"),
  "matchday 11 exact score is not stored, not a fake miss",
);
assert(settledMatches.every((match) => gradeMatch(match).actualText != null), "each settled row has an actual score");
assert(resultHitSummary(settledMatches).match_count === 10, "result hits use all 10 settled matches");
assert(overUnderSummary(settledMatches).match_count === 10, "over/under uses all 10 settled matches");
assert(bothTeamsSummary(settledMatches).match_count === 10, "both-teams-to-score uses all 10 settled matches");
const gap = runningBookGap(settledMatches);
assert(gap.length === 1 && gap[0].match_count === 10, "the running gap keeps every settled match");
assert(
  Math.abs(gap[0].gap - scorecard.book_comparison.gap_model_minus_closing) < 1e-9,
  "the running gap matches the scorecard gap",
);

const postponed = {
  ...settledMatches[0],
  match_id: 1,
  status: "postponed",
  home_goals: null,
  away_goals: null,
  result: null,
  most_likely_score: { home: 1, away: 1, probability: 0.12 },
};
const withPostponed = exactScoreSummary([...settledMatches, postponed]);
assert(withPostponed.match_count === 0, "a postponed match stays out of the exact-score count");
assert(scorecard.counts.postponed === 0, "scorecard prints the postponed count");
assert(scorecard.counts.settled === 10, "scorecard prints the settled count");
assert(scorecard.counts.pending === 0, "scorecard prints the pending count");
assert(scorecard.exact_score.match_count === 0, "exact-score tile prints its match count");
assert(typeof scorecard.exact_score.always_1_1_hits === "number", "always 1-1 sits on the same tile");
assert(
  "mean_printed_probability" in scorecard.exact_score,
  "average printed probability sits on the same tile",
);
for (const block of ["home_draw_away", "book_comparison", "exact_score", "goal_error"]) {
  assert(typeof scorecard[block].match_count === "number", `${block} prints a match count`);
}
assert(!JSON.stringify(scorecard).includes("last 10"), "no last-10 tile in the scorecard");
assert(scorecard.note.includes("small number of matches"), "the short-record sentence is on the scorecard");

assert(sampleAllowed(undefined) === false, "sample stays off unless the flag is exactly 1");
assert(sampleAllowed("") === false, "an empty sample flag does not show sample");
assert(sampleAllowed("1") === true, "PUBLIC_SHOW_SAMPLE=1 is the only sample switch");

const banned = [
  [/\bstreak\b/i, "streak"],
  [/\bunits\b/i, "units"],
  [/\bROI\b/, "ROI"],
  [/\bvalue\b/i, "value"],
  [/won the week/i, "won the week"],
  [/·/, "mid-dot"],
];

function visibleText(file, text) {
  if (!file.endsWith(".html")) return text;
  return text
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

function scan(file) {
  const text = visibleText(file, fs.readFileSync(file, "utf8"));
  for (const [pattern, name] of banned) {
    if (pattern.test(text)) {
      assert(false, `${path.relative(root, file)} contains banned ${name}`);
    }
  }
}

function walk(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (entry.name.endsWith(".html")) scan(full);
  }
}

function walkAstro(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkAstro(full);
    else if (entry.name.endsWith(".astro")) scan(full);
  }
}
walkAstro(path.join(root, "src/pages"));
walk(path.join(root, "dist"));
const distIndex = path.join(root, "dist/index.html");
if (fs.existsSync(distIndex)) {
  const html = visibleText(distIndex, fs.readFileSync(distIndex, "utf8"));
  assert(html.includes("Nott'm Forest"), "production home shows the real Forest fixture");
  assert(html.includes("Most likely score"), "production home shows a predicted score");
  assert(!html.includes("sample-local"), "production home has no sample rows");
  assert(!html.includes("127.0.0.1"), "production home does not call localhost");
  const past = fs.readFileSync(path.join(root, "dist/past/index.html"), "utf8");
  assert(past.includes("Actual"), "past page shows actual scores");
  assert(past.includes("Exact score not stored"), "past page says when the predicted score was not stored");
  assert(past.includes("postponed"), "past page shows the postponed count");
  assert(!past.includes("sample-local"), "past page has no sample rows");
  assert(!past.includes("last 10") && !past.includes("last ten"), "past page has no last-10 tile");
} else {
  assert(false, "dist is missing; run the site build before this check");
}

if (failed) {
  process.exit(1);
}
console.log("All ledger page checks passed.");

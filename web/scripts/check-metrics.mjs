/**
 * Sanity checks for web/src/lib/metrics.ts — imports the shared module
 * (no duplicated formulae) via Node's TypeScript strip-types.
 *
 * Run: npm run check:metrics
 */

import {
  topOneXTwo,
  matchBrier,
  pctWidths,
} from "../src/lib/metrics.ts";

let failed = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    failed += 1;
  } else {
    console.log("ok:", msg);
  }
}

assert(topOneXTwo({ home: 0.4, draw: 0.4, away: 0.2 }) === "H", "tie H vs D prefers H");
assert(topOneXTwo({ home: 0.3, draw: 0.35, away: 0.35 }) === "D", "tie D vs A prefers D");
assert(
  Math.abs(matchBrier({ home: 1, draw: 0, away: 0 }, "H") - 0) < 1e-12,
  "perfect home Brier is 0",
);
assert(
  Math.abs(matchBrier({ home: 0, draw: 0, away: 1 }, "H") - 2 / 3) < 1e-12,
  "wrong sure-away Brier is 2/3",
);
const w = pctWidths({ home: 0.333, draw: 0.333, away: 0.334 });
assert(w.home + w.draw + w.away === 100, "pctWidths sums to 100");

if (failed) {
  process.exit(1);
}
console.log("All metric checks passed.");

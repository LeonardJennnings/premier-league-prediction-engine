/**
 * Quick sanity checks for web/src/lib/metrics.ts (run: node --experimental-strip-types
 * is unavailable for .ts imports here — duplicate the critical asserts inline).
 *
 * Prefer: npm run check:metrics (after build) or rely on `astro build`.
 * This file mirrors the tie-break + Brier / 3 convention for CI-less local checks.
 */

function topOneXTwo(home, draw, away) {
  let best = "H";
  let bestVal = home;
  if (draw > bestVal) {
    best = "D";
    bestVal = draw;
  }
  if (away > bestVal) best = "A";
  return best;
}

function matchBrier(p, result) {
  const y = result === "H" ? [1, 0, 0] : result === "D" ? [0, 1, 0] : [0, 0, 1];
  const pred = [p.home, p.draw, p.away];
  let sum = 0;
  for (let i = 0; i < 3; i++) sum += (pred[i] - y[i]) ** 2;
  return sum / 3;
}

function pctWidths(p) {
  const raw = [
    Math.round(p.home * 100),
    Math.round(p.draw * 100),
    Math.round(p.away * 100),
  ];
  const sum = raw[0] + raw[1] + raw[2];
  if (sum === 100) return raw;
  const adjusted = [...raw];
  const maxIdx = adjusted.indexOf(Math.max(...adjusted));
  adjusted[maxIdx] += 100 - sum;
  return adjusted;
}

let failed = 0;
function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    failed += 1;
  } else {
    console.log("ok:", msg);
  }
}

assert(topOneXTwo(0.4, 0.4, 0.2) === "H", "tie H vs D prefers H");
assert(topOneXTwo(0.3, 0.35, 0.35) === "D", "tie D vs A prefers D");
assert(
  Math.abs(matchBrier({ home: 1, draw: 0, away: 0 }, "H") - 0) < 1e-12,
  "perfect home Brier is 0",
);
assert(
  Math.abs(matchBrier({ home: 0, draw: 0, away: 1 }, "H") - 2 / 3) < 1e-12,
  "wrong sure-away Brier is 2/3",
);
const w = pctWidths({ home: 0.333, draw: 0.333, away: 0.334 });
assert(w[0] + w[1] + w[2] === 100, "pctWidths sums to 100");

if (failed) {
  process.exit(1);
}
console.log("All metric checks passed.");

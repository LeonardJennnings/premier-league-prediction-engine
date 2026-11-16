/**
 * How-it-works counted peeks — same honesty as Past:
 * store-derived settled summaries, or labelled sample when empty.
 */
import { useEffect, useState } from "react";
import type { MatchPrediction } from "../lib/types";
import { buildWaffle, properScores } from "../lib/metrics";
import CountedStat from "./CountedStat";

type Props = {
  apiBase: string;
  initialMatches: MatchPrediction[];
  usingSample: boolean;
};

export default function MethodPeeks({
  apiBase,
  initialMatches,
  usingSample: initialSample,
}: Props) {
  const [matches, setMatches] = useState(initialMatches);
  const [usingSample, setUsingSample] = useState(initialSample);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = apiBase.replace(/\/$/, "");

    async function refresh() {
      try {
        const weeksRes = await fetch(`${base}/v1/predictions/weeks`, {
          headers: { Accept: "application/json" },
        });
        if (!weeksRes.ok) return;
        const weeksData = (await weeksRes.json()) as {
          weeks: { iso_week: string }[];
        };
        if (cancelled || !weeksData.weeks.length) return;

        const batches = await Promise.all(
          weeksData.weeks.map(async (w) => {
            const r = await fetch(
              `${base}/v1/predictions/weeks/${encodeURIComponent(w.iso_week)}`,
              { headers: { Accept: "application/json" } },
            );
            if (!r.ok) return [] as MatchPrediction[];
            const body = (await r.json()) as { matches: MatchPrediction[] };
            return body.matches ?? [];
          }),
        );
        if (cancelled) return;
        const all = batches.flat();
        if (all.length === 0) return;
        setMatches(all);
        setUsingSample(false);
        setRefreshNote("Method peeks refreshed from settled store.");
      } catch {
        /* keep build-time peeks */
      }
    }

    void refresh();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const scores = properScores(matches);
  const waffle = buildWaffle(matches);
  const suffix = usingSample ? " — sample" : "";
  const hitLabel =
    waffle.total > 0
      ? `top-call hits / ${waffle.total} settled${suffix}`
      : `top-call hits — none settled yet`;

  return (
    <div className="method-peeks">
      {usingSample && (
        <p className="notice" role="status">
          <strong>Note.</strong> Settled store empty or unreachable at build —
          showing labelled sample peeks, not walk-forward fold CSVs.
        </p>
      )}
      {refreshNote && (
        <p
          role="status"
          style={{ margin: "0 0 12px", fontSize: "0.72rem", color: "var(--ink-soft)" }}
        >
          {refreshNote}
        </p>
      )}
      <div className="counted">
        <CountedStat
          value={scores.meanBrier}
          decimals={3}
          label={`mean Brier (1X2)${suffix}`}
        />
        <CountedStat value={waffle.hits} label={hitLabel} />
      </div>
    </div>
  );
}

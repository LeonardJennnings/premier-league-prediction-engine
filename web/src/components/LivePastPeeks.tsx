/**
 * Past peeks island — SSRs store-derived (or labelled sample) peeks, then
 * refreshes settled rows from the API after hydrate.
 */
import { useEffect, useState } from "react";
import type { MatchPrediction } from "../lib/types";
import {
  buildWaffle,
  homeOutcomeBuckets,
  properScores,
} from "../lib/metrics";
import CountedStat from "./CountedStat";
import HitMissWaffle from "./HitMissWaffle";
import CalibrationChart from "./CalibrationChart";
import OutcomeBucketTable from "./OutcomeBucketTable";

type Props = {
  apiBase: string;
  initialPeekMatches: MatchPrediction[];
  usingSamplePeeks: boolean;
};

export default function LivePastPeeks({
  apiBase,
  initialPeekMatches,
  usingSamplePeeks: initialSample,
}: Props) {
  const [peekMatches, setPeekMatches] = useState(initialPeekMatches);
  const [usingSample, setUsingSample] = useState(initialSample);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = apiBase.replace(/\/$/, "");
    if (!base) return;

    async function refresh() {
      try {
        const weeksRes = await fetch(`${base}/v1/predictions/weeks`, {
          headers: { Accept: "application/json" },
        });
        if (!weeksRes.ok) {
          if (!cancelled) {
            setRefreshNote(
              `Live past refresh skipped (API ${weeksRes.status}).`,
            );
          }
          return;
        }
        const weeksData = (await weeksRes.json()) as {
          weeks: { iso_week: string; match_count: number }[];
        };
        if (cancelled) return;
        if (!weeksData.weeks.length) {
          setRefreshNote(
            "Live API has no settled weeks yet — keeping build-time peeks.",
          );
          return;
        }

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
        if (all.length === 0) {
          setRefreshNote(
            "Live weeks returned no matches — keeping build-time peeks.",
          );
          return;
        }

        setPeekMatches(all);
        setUsingSample(false);
        setRefreshNote("Past peeks refreshed from live API after load.");
      } catch {
        if (!cancelled) {
          setRefreshNote(
            "Live past refresh unreachable — keeping build-time peeks.",
          );
        }
      }
    }

    void refresh();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const waffle = buildWaffle(peekMatches);
  const scores = properScores(peekMatches);
  const buckets = homeOutcomeBuckets(peekMatches);
  const brierLabel = usingSample
    ? "mean Brier (1X2) — sample"
    : "mean Brier (1X2)";
  const llLabel = usingSample
    ? "mean log loss — sample"
    : "mean log loss (1X2)";

  return (
    <section className="peeks" aria-label="Past peeks">
      {refreshNote && (
        <p className="refresh-note" role="status">
          {refreshNote}
        </p>
      )}
      <div className="peek">
        <h2 className="section">Past / hit vs miss</h2>
        <div className="counted">
          <CountedStat
            value={waffle.hits}
            label={`hits / ${waffle.total} settled`}
          />
          <CountedStat
            value={scores.meanBrier}
            decimals={3}
            label={brierLabel}
          />
          <CountedStat
            value={scores.meanLogLoss}
            decimals={3}
            label={llLabel}
          />
        </div>
        <HitMissWaffle cells={waffle.cells} />
      </div>
      <div className="peek">
        <h2 className="section">Calibration peek</h2>
        <CalibrationChart points={buckets.points} />
        <div className="bucket-wrap">
          <OutcomeBucketTable rows={buckets.rows} />
        </div>
      </div>
      <style>{`
        .peeks {
          display: grid;
          grid-template-columns: 1.15fr 0.85fr;
          gap: 22px;
          margin-bottom: 24px;
          padding-top: 8px;
          border-top: 1px solid var(--line);
        }
        .counted {
          display: flex;
          flex-wrap: wrap;
          gap: 20px;
          margin-bottom: 14px;
        }
        .bucket-wrap {
          margin-top: 16px;
        }
        .refresh-note {
          margin: 0 0 12px;
          font-size: 0.72rem;
          color: var(--ink-soft);
          grid-column: 1 / -1;
        }
        @media (max-width: 640px) {
          .peeks {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </section>
  );
}

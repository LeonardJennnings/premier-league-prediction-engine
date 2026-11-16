/**
 * Weekend slate island — SSRs the build-time slate, then lightly refreshes
 * from PUBLIC_PREDICTIONS_API_BASE after hydrate so dist is not forever sample-only.
 */
import { useEffect, useState } from "react";
import type { MatchPrediction } from "../lib/types";
import { formatKickoff, secondaryMarketsLine } from "../lib/format";
import { pctWidths } from "../lib/metrics";

type Props = {
  apiBase: string;
  initialMatches: MatchPrediction[];
  initialModelVersion: string;
  initialNotice: string | null;
};

type StripProps = {
  kind: "model" | "book";
  home: number | null;
  draw: number | null;
  away: number | null;
};

function ProbabilityStrip({ kind, home, draw, away }: StripProps) {
  const missing = home == null || draw == null || away == null;
  const widths = missing
    ? { home: 0, draw: 0, away: 0 }
    : pctWidths({ home: home!, draw: draw!, away: away! });
  const label = kind === "model" ? "Model" : "Book";
  const aria = missing
    ? `${label} probabilities unavailable`
    : `${label} ${widths.home} home, ${widths.draw} draw, ${widths.away} away`;

  return (
    <div className="strip-row">
      <span className={`strip-label${kind === "model" ? " model" : ""}`}>
        {label}
      </span>
      {missing ? (
        <div className="bar empty" role="img" aria-label={aria}>
          <span className="empty-copy">No book line</span>
        </div>
      ) : (
        <div
          className={`bar ${kind === "model" ? "model-bar" : "book-bar"}`}
          role="img"
          aria-label={aria}
        >
          <div className="seg home" style={{ width: `${widths.home}%` }}>
            {widths.home >= 12 ? widths.home : ""}
          </div>
          <div className="seg draw" style={{ width: `${widths.draw}%` }}>
            {widths.draw >= 12 ? widths.draw : ""}
          </div>
          <div className="seg away" style={{ width: `${widths.away}%` }}>
            {widths.away >= 12 ? widths.away : ""}
          </div>
        </div>
      )}
      <span className="hd-a nums">
        {missing
          ? "H — / D — / A —"
          : `H ${widths.home} / D ${widths.draw} / A ${widths.away}`}
      </span>
    </div>
  );
}

function MatchCard({ match }: { match: MatchPrediction }) {
  return (
    <article className="match">
      <div className="fixture">
        <h3>
          {match.home_team} vs {match.away_team}
        </h3>
        <span className="kick nums">{formatKickoff(match.kickoff_utc)}</span>
      </div>
      <div className="strips">
        <ProbabilityStrip
          kind="model"
          home={match.p_home}
          draw={match.p_draw}
          away={match.p_away}
        />
        <ProbabilityStrip
          kind="book"
          home={match.book_p_home}
          draw={match.book_p_draw}
          away={match.book_p_away}
        />
      </div>
      <p className="secondary nums">{secondaryMarketsLine(match)}</p>
    </article>
  );
}

export default function LiveWeekend({
  apiBase,
  initialMatches,
  initialModelVersion,
  initialNotice,
}: Props) {
  const [matches, setMatches] = useState(initialMatches);
  const [modelVersion, setModelVersion] = useState(initialModelVersion);
  const [notice, setNotice] = useState(initialNotice);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const base = apiBase.replace(/\/$/, "");

    async function refresh() {
      try {
        const res = await fetch(`${base}/v1/predictions/upcoming`, {
          headers: { Accept: "application/json" },
        });
        if (!res.ok) {
          if (!cancelled) {
            setRefreshNote(
              `Live refresh skipped (API ${res.status}). Showing build-time slate.`,
            );
          }
          return;
        }
        const data = (await res.json()) as {
          model_version: string;
          matches: MatchPrediction[];
        };
        if (cancelled) return;
        if (!data.matches || data.matches.length === 0) {
          setRefreshNote("Live API returned an empty slate — keeping build-time rows.");
          return;
        }
        setMatches(data.matches);
        setModelVersion(data.model_version);
        setNotice(null);
        setRefreshNote("Refreshed from live API after load.");
      } catch {
        if (!cancelled) {
          setRefreshNote(
            "Live refresh unreachable — keeping build-time slate (sample if API was down at build).",
          );
        }
      }
    }

    void refresh();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  return (
    <div className="match-slate">
      {notice && (
        <p className="notice" role="status">
          <strong>Note.</strong> {notice}
        </p>
      )}
      {refreshNote && (
        <p className="refresh-note" role="status">
          {refreshNote}
        </p>
      )}
      <p className="meta-line nums">
        Model <code>{modelVersion}</code>
        {" — "}
        API <code>{apiBase}</code>
      </p>
      {matches.map((m) => (
        <MatchCard key={m.match_id} match={m} />
      ))}
    </div>
  );
}

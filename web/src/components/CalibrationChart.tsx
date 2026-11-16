/**
 * Calibration / reliability curve for home-win probability.
 * Diagonal = perfect calibration. Points from settled outcome buckets.
 */

export type CalibrationPointProp = {
  predicted: number;
  observed: number;
  n: number;
};

type Props = {
  points?: CalibrationPointProp[];
  note?: string;
};

export default function CalibrationChart({
  points = [],
  note = "Reliability peek for home wins: points near the diagonal are well calibrated.",
}: Props) {
  const w = 240;
  const h = 140;
  const padL = 24;
  const padB = 24;
  const padT = 16;
  const padR = 20;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const toX = (p: number) => padL + p * innerW;
  const toY = (p: number) => padT + (1 - p) * innerH;

  const sorted = [...points].sort((a, b) => a.predicted - b.predicted);
  const poly =
    sorted.length === 0
      ? ""
      : sorted
          .map((pt, i) => {
            const x = toX(pt.predicted).toFixed(1);
            const y = toY(pt.observed).toFixed(1);
            return `${i === 0 ? "" : " "}${x},${y}`;
          })
          .join("");

  return (
    <figure className="cal">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={
          sorted.length === 0
            ? "Calibration curve unavailable"
            : "Reliability curve of predicted versus observed home-win rate"
        }
      >
        <line
          x1={padL}
          y1={h - padB}
          x2={w - padR}
          y2={h - padB}
          stroke="currentColor"
          strokeWidth="1"
        />
        <line
          x1={padL}
          y1={h - padB}
          x2={padL}
          y2={padT}
          stroke="currentColor"
          strokeWidth="1"
        />
        <line
          x1={padL}
          y1={h - padB}
          x2={w - padR}
          y2={padT}
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="3 3"
          opacity="0.45"
        />
        {poly ? (
          <polyline
            fill="none"
            stroke="var(--model)"
            strokeWidth="2"
            points={poly}
          />
        ) : null}
        {sorted.map((pt) => (
          <circle
            key={`${pt.predicted}-${pt.n}`}
            cx={toX(pt.predicted)}
            cy={toY(pt.observed)}
            r={3.2}
            fill="var(--model)"
          >
            <title>
              Predicted {Math.round(pt.predicted * 100)}%, observed{" "}
              {Math.round(pt.observed * 100)}% (n={pt.n})
            </title>
          </circle>
        ))}
      </svg>
      <figcaption>
        {sorted.length === 0
          ? "No settled matches yet for a reliability curve."
          : note}
      </figcaption>
      <style>{`
        .cal {
          margin: 0;
          color: var(--ink-soft);
        }
        .cal svg {
          width: 100%;
          max-width: 320px;
          height: auto;
          display: block;
          background: var(--paper-2);
          border: 1px solid var(--line-soft);
          border-radius: 2px;
        }
        .cal figcaption {
          margin-top: 8px;
          font-size: 0.75rem;
        }
      `}</style>
    </figure>
  );
}

/**
 * Calibration / reliability curve shell.
 * TODO(Data Scientist): plot reliability points from walk-forward folds;
 * soft draw-in optional via Designer + Motion.
 */

type Props = {
  note?: string;
};

export default function CalibrationChart({
  note = "Lorem ipsum dolor sit amet — curve placeholder until fold metrics land.",
}: Props) {
  return (
    <figure className="cal">
      <svg
        viewBox="0 0 240 140"
        role="img"
        aria-label="Calibration curve placeholder"
      >
        <line x1="24" y1="116" x2="220" y2="116" stroke="currentColor" strokeWidth="1" />
        <line x1="24" y1="116" x2="24" y2="16" stroke="currentColor" strokeWidth="1" />
        <line
          x1="24"
          y1="116"
          x2="220"
          y2="16"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="3 3"
          opacity="0.45"
        />
        {/* Placeholder polyline — DS replaces with real points */}
        <polyline
          fill="none"
          stroke="var(--model)"
          strokeWidth="2"
          points="24,110 60,96 100,78 140,62 180,40 220,28"
        />
      </svg>
      <figcaption>{note}</figcaption>
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

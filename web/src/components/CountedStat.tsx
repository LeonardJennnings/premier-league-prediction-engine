"use client";

/**
 * Peek metric. The first paint is the real figure, so a static page
 * and a reader with scripts off do not see a zero.
 */

type Props = {
  value: number | null;
  label: string;
  decimals?: number;
};

export default function CountedStat({ value, label, decimals = 0 }: Props) {
  const display = value;

  const formatted =
    display == null
      ? "—"
      : decimals > 0
        ? display.toFixed(decimals)
        : String(Math.round(display));

  return (
    <div className="stat">
      <strong className="nums">{formatted}</strong>
      <span>{label}</span>
      <style>{`
        .stat strong {
          display: block;
          font-size: 1.85rem;
          font-weight: 700;
          letter-spacing: -0.03em;
          font-variant-numeric: tabular-nums;
          line-height: 1;
          font-family: var(--font-sans);
        }
        .stat span {
          display: block;
          margin-top: 4px;
          font-size: 0.92rem;
          color: var(--ink-soft);
        }
      `}</style>
    </div>
  );
}

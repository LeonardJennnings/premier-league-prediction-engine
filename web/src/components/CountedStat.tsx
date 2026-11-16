/**
 * Counted summary number shell.
 * TODO(Designer): wire Motion `whileInView` count-up (~0.6–1.2s ease-out);
 * respect prefers-reduced-motion (show final value immediately).
 */

type Props = {
  value: string;
  label: string;
};

export default function CountedStat({ value, label }: Props) {
  return (
    <div className="stat">
      <strong className="nums">{value}</strong>
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
          font-size: 0.72rem;
          color: var(--ink-soft);
        }
      `}</style>
    </div>
  );
}

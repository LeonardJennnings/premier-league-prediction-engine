/**
 * Outcome-bucket frequency table shell (viz #6).
 * TODO(Data Scientist): bind real walk-forward buckets
 * (when model said ~X%, what happened). Keep distinct from hit/miss waffle.
 * TODO(Designer): HeroUI Table theming if adopted.
 */

type BucketRow = {
  bucket: string;
  predictedShare: string;
  observedShare: string;
  n: string;
};

type Props = {
  rows?: BucketRow[];
};

const PLACEHOLDER: BucketRow[] = [
  { bucket: "30–40%", predictedShare: "lorem", observedShare: "ipsum", n: "—" },
  { bucket: "40–50%", predictedShare: "dolor", observedShare: "sit", n: "—" },
  { bucket: "50–60%", predictedShare: "amet", observedShare: "consectetur", n: "—" },
];

export default function OutcomeBucketTable({ rows = PLACEHOLDER }: Props) {
  return (
    <div className="buckets">
      <table>
        <caption>Outcome buckets (placeholder)</caption>
        <thead>
          <tr>
            <th scope="col">Model said</th>
            <th scope="col">Share predicted</th>
            <th scope="col">Share observed</th>
            <th scope="col">n</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.bucket}>
              <th scope="row">{row.bucket}</th>
              <td className="nums">{row.predictedShare}</td>
              <td className="nums">{row.observedShare}</td>
              <td className="nums">{row.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <style>{`
        .buckets table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.8rem;
        }
        .buckets caption {
          text-align: left;
          font-size: 0.72rem;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: var(--ink-soft);
          margin-bottom: 8px;
        }
        .buckets th,
        .buckets td {
          border-top: 1px solid var(--line-soft);
          padding: 6px 4px;
          text-align: left;
        }
        .buckets thead th {
          border-top: none;
          color: var(--ink-soft);
          font-weight: 500;
        }
      `}</style>
    </div>
  );
}

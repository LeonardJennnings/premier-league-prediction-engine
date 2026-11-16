/**
 * Outcome-bucket frequency table (viz #6).
 * When the model said home ~X%, what happened — distinct from hit/miss waffle.
 */

type BucketRow = {
  bucket: string;
  predictedShare: string;
  observedShare: string;
  n: string;
};

type Props = {
  rows?: BucketRow[];
  caption?: string;
};

export default function OutcomeBucketTable({
  rows = [],
  caption = "Home-win probability bands vs observed home-win rate",
}: Props) {
  if (rows.length === 0) {
    return (
      <div className="buckets">
        <p className="empty">No outcome buckets yet — need settled matches.</p>
        <style>{`
          .buckets .empty {
            margin: 0;
            font-size: 0.8rem;
            color: var(--ink-soft);
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="buckets">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Model P(home)</th>
            <th scope="col">Mean predicted</th>
            <th scope="col">Home won</th>
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

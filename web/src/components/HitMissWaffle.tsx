/**
 * Hit / miss waffle — one cell per settled match.
 * Hit = model's top 1X2 matched the result. Labels required (not colour-only).
 * Version C peek chrome.
 */

export type WaffleCellProp = boolean | { hit: boolean; title?: string };

type Props = {
  cells: WaffleCellProp[];
  caption?: string;
};

function normalize(cell: WaffleCellProp): { hit: boolean; title: string } {
  if (typeof cell === "boolean") {
    return { hit: cell, title: cell ? "Hit" : "Miss" };
  }
  return {
    hit: cell.hit,
    title: cell.title ?? (cell.hit ? "Hit" : "Miss"),
  };
}

export default function HitMissWaffle({
  cells,
  caption = "Each square is one settled match. Hit = model’s top home / draw / away matched the result. Miss = it did not.",
}: Props) {
  const normalized = cells.map(normalize);
  const hits = normalized.filter((c) => c.hit).length;
  const total = normalized.length;

  if (total === 0) {
    return (
      <div className="waffle-block">
        <p className="waffle-caption">
          No settled matches yet — the waffle appears after results land.
        </p>
      </div>
    );
  }

  return (
    <div className="waffle-block">
      <p className="waffle-caption">
        {caption}{" "}
        <em>
          {hits} hits / {total} settled
        </em>
      </p>
      <div
        className="waffle"
        role="img"
        aria-label={`Waffle: ${hits} hits and ${total - hits} misses across ${total} settled matches`}
      >
        {normalized.map((c, i) => (
          <div
            key={i}
            className={c.hit ? "cell hit" : "cell miss"}
            title={c.title}
          />
        ))}
      </div>
      <div className="legend">
        <span>
          <i className="sw-hit" aria-hidden="true" />
          Hit ({hits})
        </span>
        <span>
          <i className="sw-miss" aria-hidden="true" />
          Miss ({total - hits})
        </span>
      </div>
      <style>{`
        .waffle-block { margin: 0; }
        .waffle-caption {
          font-size: 0.8rem;
          color: var(--ink);
          margin: 0 0 10px;
        }
        .waffle-caption em {
          font-style: normal;
          font-weight: 600;
          color: var(--model);
        }
        .waffle {
          display: grid;
          grid-template-columns: repeat(10, 1fr);
          gap: 3px;
          max-width: 280px;
        }
        .cell {
          aspect-ratio: 1;
          border-radius: 1px;
          background: var(--miss);
          border: 1px solid var(--line);
        }
        .cell.hit {
          background: var(--hit);
          border-color: var(--hit);
        }
        .legend {
          display: flex;
          gap: 14px;
          margin-top: 10px;
          font-size: 0.72rem;
          color: var(--ink-soft);
        }
        .legend i {
          display: inline-block;
          width: 10px;
          height: 10px;
          margin-right: 5px;
          vertical-align: -1px;
          border-radius: 1px;
          border: 1px solid var(--line);
        }
        .sw-hit { background: var(--hit); border-color: var(--hit); }
        .sw-miss { background: var(--miss); }
      `}</style>
    </div>
  );
}

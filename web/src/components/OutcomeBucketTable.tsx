"use client";

/**
 * Outcome-bucket frequency table (viz #6) — HeroUI Table, Version C themed.
 * When the model said home ~X%, what happened — distinct from hit/miss waffle.
 */

import { Table } from "@heroui/react";

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
      <p className="buckets-caption">{caption}</p>
      <Table variant="secondary">
        <Table.ScrollContainer>
          <Table.Content aria-label={caption}>
            <Table.Header>
              <Table.Column isRowHeader>Model P(home)</Table.Column>
              <Table.Column>Mean predicted</Table.Column>
              <Table.Column>Home won</Table.Column>
              <Table.Column>n</Table.Column>
            </Table.Header>
            <Table.Body>
              {rows.map((row) => (
                <Table.Row key={row.bucket}>
                  <Table.Cell>{row.bucket}</Table.Cell>
                  <Table.Cell>
                    <span className="nums">{row.predictedShare}</span>
                  </Table.Cell>
                  <Table.Cell>
                    <span className="nums">{row.observedShare}</span>
                  </Table.Cell>
                  <Table.Cell>
                    <span className="nums">{row.n}</span>
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Content>
        </Table.ScrollContainer>
      </Table>
      <style>{`
        .buckets {
          font-size: 0.8rem;
        }
        .buckets-caption {
          margin: 0 0 8px;
          font-size: 0.72rem;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: var(--ink-soft);
        }
        .buckets .table,
        .buckets [class*="table"] {
          --border: var(--line-soft);
          background: transparent;
        }
        .buckets .nums {
          font-variant-numeric: tabular-nums;
          font-family: var(--font-mono);
        }
      `}</style>
    </div>
  );
}

"use client";

/**
 * Version C counted peek metric — Motion scroll count-up.
 * Ease-out ~0.9s once in view; prefers-reduced-motion → final value immediately.
 */

import { animate, useInView, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

type Props = {
  value: number | null;
  label: string;
  decimals?: number;
};

export default function CountedStat({ value, label, decimals = 0 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduceMotion = useReducedMotion();
  const [display, setDisplay] = useState(() =>
    value == null || reduceMotion ? value : 0,
  );

  useEffect(() => {
    if (!inView) return;
    if (value == null) {
      setDisplay(null);
      return;
    }
    if (reduceMotion) {
      setDisplay(value);
      return;
    }
    const controls = animate(0, value, {
      duration: 0.9,
      ease: "easeOut",
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [inView, reduceMotion, value]);

  const formatted =
    display == null
      ? "—"
      : decimals > 0
        ? display.toFixed(decimals)
        : String(Math.round(display));

  return (
    <div className="stat" ref={ref}>
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
          font-size: 0.72rem;
          color: var(--ink-soft);
        }
      `}</style>
    </div>
  );
}

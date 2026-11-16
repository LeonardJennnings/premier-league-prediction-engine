/** Display helpers — percentages as whole numbers; tabular-friendly strings. */

export function pct(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return String(Math.round(p * 100));
}

export function pctLabel(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${Math.round(p * 100)}%`;
}

export function formatKickoff(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Europe/London",
  }).format(d);
}

export function resultLabel(result: string | null | undefined): string {
  if (!result) return "—";
  if (result === "H") return "Home";
  if (result === "D") return "Draw";
  if (result === "A") return "Away";
  return result;
}

/** Top 1X2 outcome from model probs — used for hit/miss shells (DS owns final binding). */
export function topOneXTwo(pHome: number, pDraw: number, pAway: number): "H" | "D" | "A" {
  if (pHome >= pDraw && pHome >= pAway) return "H";
  if (pDraw >= pAway) return "D";
  return "A";
}

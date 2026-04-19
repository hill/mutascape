import type { ChoiceSite } from "./types";

export interface OracleMeta {
  oracle: boolean;
  datasets: string[];
  hard_max_k: number;
}

export interface OracleScore {
  mask: number;
  fitness: number;
}

export interface OracleResponse {
  backend: "gb1-real" | "synthetic";
  scores: OracleScore[];
}

/** Probe for a live oracle at /api/meta. Resolves to null if unreachable so
 * the caller can fall back to the client-side synthetic model. */
export async function probeOracle(): Promise<OracleMeta | null> {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 800);
    const res = await fetch("./api/meta", { signal: ctl.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    return (await res.json()) as OracleMeta;
  } catch {
    return null;
  }
}

export async function scoreViaOracle(
  wt: string,
  sites: ChoiceSite[],
  alpha: number,
  seed: number,
  masks: number[],
): Promise<OracleResponse> {
  const res = await fetch("./api/score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      wt,
      sites: sites.map((s) => ({ position: s.position, altAA: s.altAA })),
      alpha,
      seed,
      masks,
    }),
  });
  if (!res.ok) throw new Error(`oracle: ${res.status}`);
  return (await res.json()) as OracleResponse;
}

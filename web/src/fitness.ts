import { Rng } from "./rng";
import type { ChoiceSite, Gb1Data } from "./types";

export interface FitnessModel {
  readonly k: number;
  alpha: number;
  score(mask: number): number;
  singleEffect(bit: number): number;
  couplingEffect(i: number, j: number): number;
}

export class SyntheticModel implements FitnessModel {
  readonly k: number;
  alpha: number;
  private h: number[];
  private J: number[][];

  constructor(
    sites: ChoiceSite[],
    alpha: number,
    rng: Rng,
    singleOverrides?: (number | null)[],
  ) {
    this.k = sites.length;
    this.alpha = alpha;

    this.h = new Array(this.k);
    for (let i = 0; i < this.k; i++) {
      const override = singleOverrides?.[i];
      this.h[i] = override != null ? override : rng.gauss();
    }

    this.J = Array.from({ length: this.k }, () => new Array(this.k).fill(0));
    for (let i = 0; i < this.k; i++) {
      for (let j = i + 1; j < this.k; j++) {
        const v = rng.gauss();
        this.J[i][j] = v;
        this.J[j][i] = v;
      }
    }
  }

  score(mask: number): number {
    let f = 0;
    for (let i = 0; i < this.k; i++) {
      if ((mask >> i) & 1) f += this.h[i];
    }
    for (let i = 0; i < this.k; i++) {
      if (!((mask >> i) & 1)) continue;
      for (let j = i + 1; j < this.k; j++) {
        if ((mask >> j) & 1) f += this.alpha * this.J[i][j];
      }
    }
    return f;
  }

  singleEffect(bit: number): number {
    return this.h[bit];
  }

  couplingEffect(i: number, j: number): number {
    return this.J[i][j];
  }
}

export function gb1SingleOverrides(
  gb1: Gb1Data,
  sites: ChoiceSite[],
): (number | null)[] {
  return sites.map((s) => {
    const row = gb1.olson.singles[String(s.position)];
    if (!row) return null;
    const v = row[s.altAA];
    return typeof v === "number" ? v : null;
  });
}

/** Return whether the current sites line up exactly with the Olson 4-site
 * combinatorial screen so we can overlay measured values as ground truth. */
export function sitesMatchOlson(gb1: Gb1Data, sites: ChoiceSite[]): boolean {
  if (sites.length !== gb1.olson.positions.length) return false;
  const positions = new Set(gb1.olson.positions);
  for (const s of sites) if (!positions.has(s.position)) return false;
  return true;
}

/** Build a per-mask lookup of measured fitness for the given masks, or null
 * when no value exists for that specific AA combination. */
export function measuredByMask(
  gb1: Gb1Data,
  sites: ChoiceSite[],
  masks: Iterable<number>,
): Map<number, number> | null {
  if (!sitesMatchOlson(gb1, sites)) return null;

  const bySite = new Map<number, ChoiceSite>();
  for (const s of sites) bySite.set(s.position, s);

  const table = new Map<number, number>();

  for (const m of masks) {
    const key = gb1.olson.positions
      .map((pos) => {
        const site = bySite.get(pos)!;
        const siteIndex = sites.indexOf(site);
        const mutated = (m >> siteIndex) & 1;
        return mutated ? site.altAA : site.wtAA;
      })
      .join("");
    const v = gb1.olson.measured[key];
    if (typeof v === "number") table.set(m, v);
  }
  return table;
}

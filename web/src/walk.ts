import type { Lattice, Vertex } from "./types";

function vert(lat: Lattice, mask: number): Vertex | undefined {
  return lat.vertices.get(mask);
}

export function localOptima(lat: Lattice): number[] {
  const optima: number[] = [];
  for (const v of lat.vertices.values()) {
    let isOpt = true;
    for (let b = 0; b < lat.k; b++) {
      const n = vert(lat, v.mask ^ (1 << b));
      if (!n) continue;
      if (n.fitness > v.fitness) {
        isOpt = false;
        break;
      }
    }
    if (isOpt) optima.push(v.mask);
  }
  return optima;
}

export function greedyWalk(lat: Lattice, startMask = 0): number[] {
  const start = vert(lat, startMask);
  if (!start) return [startMask];

  const path = [startMask];
  let cur = startMask;
  const visited = new Set<number>([cur]);
  while (true) {
    let best = -1;
    let bestF = vert(lat, cur)!.fitness;
    for (let b = 0; b < lat.k; b++) {
      const nm = cur ^ (1 << b);
      if (visited.has(nm)) continue;
      const nb = vert(lat, nm);
      if (!nb) continue;
      if (nb.fitness > bestF) {
        bestF = nb.fitness;
        best = nm;
      }
    }
    if (best < 0) break;
    path.push(best);
    visited.add(best);
    cur = best;
  }
  return path;
}

/** Steepest-ascending ordering of the bits to flip when travelling from WT to
 * `target`: at each step pick the single mutation that yields the highest
 * fitness so far. Makes the click-to-trace animation read like a small
 * adaptive walk toward the chosen variant. */
export function shortestPathFromWT(lat: Lattice, target: number): number[] {
  const bits: number[] = [];
  for (let b = 0; b < lat.k; b++) if ((target >> b) & 1) bits.push(b);
  const path = [0];
  let cur = 0;
  const remaining = new Set(bits);
  while (remaining.size > 0) {
    let bestBit = -1;
    let bestF = -Infinity;
    for (const b of remaining) {
      const nm = cur | (1 << b);
      const v = vert(lat, nm);
      const f = v ? v.fitness : -Infinity;
      if (f > bestF) {
        bestF = f;
        bestBit = b;
      }
    }
    if (bestBit < 0) break;
    cur |= 1 << bestBit;
    remaining.delete(bestBit);
    path.push(cur);
  }
  return path;
}

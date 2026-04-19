import type { ChoiceSite, Edge, Lattice, Vertex } from "./types";

/** Largest k for which we materialize every 2^k vertex. Beyond this we switch
 * to a Hamming-ball around the cursor to keep render + hit-test snappy. */
const DENSE_K_LIMIT = 9;
/** Max number of vertices we try to keep visible in sparse mode. The focus
 * radius is picked so the ball fits under this budget. */
const SPARSE_BUDGET = 280;

function popcount(x: number): number {
  x -= (x >> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let num = 1;
  for (let i = 0; i < k; i++) num = (num * (n - i)) / (i + 1);
  return Math.round(num);
}

function chooseFocusRadius(k: number): number {
  let total = 0;
  for (let r = 0; r <= k; r++) {
    const add = binomial(k, r);
    if (total + add > SPARSE_BUDGET) return Math.max(1, r === 0 ? 1 : r - 1 > 0 ? r - 1 : 1);
    total += add;
  }
  return k;
}

/** Walk every subset of [0..k) of cardinality in [0..radius], XOR the bits
 * into `center`, and emit the resulting mask. */
function enumerateBall(
  center: number,
  k: number,
  radius: number,
  emit: (mask: number) => void,
) {
  emit(center);
  for (let r = 1; r <= radius; r++) {
    const idx: number[] = Array.from({ length: r }, (_, i) => i);
    while (true) {
      let m = center;
      for (const b of idx) m ^= 1 << b;
      emit(m);
      let i = r - 1;
      while (i >= 0 && idx[i] === k - r + i) i--;
      if (i < 0) break;
      idx[i]++;
      for (let j = i + 1; j < r; j++) idx[j] = idx[j - 1] + 1;
    }
  }
}

export interface BuildOpts {
  cursorMask?: number;
  forceDense?: boolean;
  radius?: number;
}

export function buildLattice(sites: ChoiceSite[], opts: BuildOpts = {}): Lattice {
  const k = sites.length;
  const cursorMask = opts.cursorMask ?? 0;
  const isDense = opts.forceDense || k <= DENSE_K_LIMIT;
  const focusRadius = isDense ? k : opts.radius ?? chooseFocusRadius(k);

  const vertices = new Map<number, Vertex>();
  const byDepth: Vertex[][] = Array.from({ length: k + 1 }, () => []);

  const addMask = (m: number) => {
    if (vertices.has(m)) return;
    const v: Vertex = {
      mask: m,
      depth: popcount(m),
      col: 0,
      fitness: 0,
      measured: null,
      x: 0,
      y: 0,
    };
    vertices.set(m, v);
    byDepth[v.depth].push(v);
  };

  if (isDense) {
    const n = 1 << k;
    for (let m = 0; m < n; m++) addMask(m);
  } else {
    enumerateBall(cursorMask, k, focusRadius, addMask);
  }

  for (let d = 0; d <= k; d++) {
    byDepth[d].sort((a, b) => a.mask - b.mask);
    byDepth[d].forEach((v, i) => (v.col = i));
  }

  const edges: Edge[] = [];
  for (const v of vertices.values()) {
    for (let b = 0; b < k; b++) {
      const nb = v.mask ^ (1 << b);
      if (nb > v.mask && vertices.has(nb)) {
        edges.push({ from: v.mask, to: nb, diffBit: b });
      }
    }
  }

  return {
    k,
    sites,
    vertices,
    edges,
    byDepth,
    isDense,
    cursorMask,
    focusRadius,
  };
}

export function layoutLattice(lat: Lattice, width: number, height: number) {
  const { k, byDepth } = lat;
  const padX = Math.max(52, width * 0.08);
  const padY = Math.max(46, height * 0.09);

  if (k === 0) return;

  const nonEmpty: number[] = [];
  for (let d = 0; d <= k; d++) {
    if (byDepth[d].length > 0) nonEmpty.push(d);
  }
  const slots = nonEmpty.length;

  for (let slot = 0; slot < slots; slot++) {
    const d = nonEmpty[slot];
    const row = byDepth[d];
    const y =
      slots === 1
        ? height / 2
        : padY + (slot * (height - 2 * padY)) / (slots - 1);

    if (row.length === 1) {
      row[0].x = width / 2;
      row[0].y = y;
      continue;
    }

    const spread = Math.min(
      width - 2 * padX,
      Math.max(180, row.length * 42),
    );
    const left = (width - spread) / 2;
    row.forEach((v, i) => {
      v.x = left + (i * spread) / (row.length - 1);
      v.y = y;
    });
  }
}

export function scoreLattice(
  lat: Lattice,
  predict: (mask: number) => number,
  measuredTable: Map<number, number> | null,
) {
  for (const v of lat.vertices.values()) {
    v.fitness = predict(v.mask);
    v.measured = measuredTable?.get(v.mask) ?? null;
  }
}

export function findVertexAt(
  lat: Lattice,
  x: number,
  y: number,
  radius: number,
): Vertex | null {
  let best: Vertex | null = null;
  let bestDist = radius * radius;
  for (const v of lat.vertices.values()) {
    const dx = v.x - x;
    const dy = v.y - y;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist) {
      bestDist = d2;
      best = v;
    }
  }
  return best;
}

export function hammingDistance(a: number, b: number): number {
  return popcount(a ^ b);
}

export { DENSE_K_LIMIT };

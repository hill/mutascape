export const AAS = "ACDEFGHIKLMNPQRSTVWY";

export interface ChoiceSite {
  position: number;
  wtAA: string;
  altAA: string;
}

export interface Vertex {
  mask: number;
  depth: number;
  col: number;
  fitness: number;
  measured: number | null;
  x: number;
  y: number;
}

export interface Edge {
  from: number;
  to: number;
  diffBit: number;
}

/** A Hamming lattice over k mutational sites. For small k the lattice is
 * fully materialized (dense); for large k only a Hamming-ball around the
 * cursor is materialized (sparse) and the UI re-centers on click. */
export interface Lattice {
  k: number;
  sites: ChoiceSite[];
  vertices: Map<number, Vertex>;
  edges: Edge[];
  byDepth: Vertex[][];
  isDense: boolean;
  cursorMask: number;
  focusRadius: number;
}

export interface Gb1Data {
  protein: "GB1";
  reference: string;
  wt: string;
  olson: {
    positions: number[];
    wt_aas: string;
    singles: Record<string, Record<string, number>>;
    measured: Record<string, number>;
  };
}

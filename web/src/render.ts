import type { Lattice, Vertex } from "./types";

const NEUTRAL: [number, number, number] = [200, 198, 192];
const BAD: [number, number, number] = [192, 66, 43];
const GOOD: [number, number, number] = [44, 74, 158];
const INK: [number, number, number] = [10, 10, 10];
const HIGHLIGHT = "#0a0a0a";

export interface RenderState {
  lattice: Lattice;
  hoveredMask: number | null;
  activePath: number[] | null;
  pathProgress: number;
  colorRange: number;
  highlightMask: number | null;
  /** When set, only these vertices (and edges between two of them) are drawn.
   * null means "render every vertex" (used for small k). */
  visibleMasks: Set<number> | null;
  /** Cursor vertex; renders with a lime ring in focus mode. */
  cursorMask: number | null;
}

function divergingColor(value: number, range: number): string {
  if (range === 0) return rgb(NEUTRAL);
  const t = Math.max(-1, Math.min(1, value / range));
  const target = t < 0 ? BAD : GOOD;
  const amt = Math.abs(t);
  const r = Math.round(NEUTRAL[0] * (1 - amt) + target[0] * amt);
  const g = Math.round(NEUTRAL[1] * (1 - amt) + target[1] * amt);
  const b = Math.round(NEUTRAL[2] * (1 - amt) + target[2] * amt);
  return `rgb(${r},${g},${b})`;
}

function rgb([r, g, b]: [number, number, number]) {
  return `rgb(${r},${g},${b})`;
}

function vertexRadius(lat: Lattice): number {
  return lat.k >= 5 ? 13 : 17;
}

function isHighlightedEdge(
  state: RenderState,
  from: number,
  to: number,
): boolean {
  if (state.hoveredMask === null) return false;
  return state.hoveredMask === from || state.hoveredMask === to;
}

function edgeInPath(state: RenderState, from: number, to: number): boolean {
  if (!state.activePath) return false;
  for (let i = 0; i < state.activePath.length - 1; i++) {
    const a = state.activePath[i];
    const b = state.activePath[i + 1];
    if ((a === from && b === to) || (a === to && b === from)) return true;
  }
  return false;
}

function drawEdges(ctx: CanvasRenderingContext2D, state: RenderState) {
  const { lattice, visibleMasks } = state;
  const verts = lattice.vertices;
  const hasHover = state.hoveredMask !== null;

  ctx.lineCap = "round";
  for (const e of lattice.edges) {
    if (visibleMasks && (!visibleMasks.has(e.from) || !visibleMasks.has(e.to))) continue;
    const a = verts.get(e.from);
    const b = verts.get(e.to);
    if (!a || !b) continue;
    const inPath = edgeInPath(state, e.from, e.to);
    const hot = isHighlightedEdge(state, e.from, e.to);

    let alpha = hasHover ? 0.08 : 0.18;
    let width = 1;
    if (hot) {
      alpha = 0.5;
      width = 1.2;
    }

    if (inPath) {
      ctx.strokeStyle = HIGHLIGHT;
      ctx.lineWidth = 2;
    } else {
      ctx.strokeStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, ${alpha})`;
      ctx.lineWidth = width;
    }
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawVertex(
  ctx: CanvasRenderingContext2D,
  v: Vertex,
  state: RenderState,
) {
  const r = vertexRadius(state.lattice);
  const isHover = state.hoveredMask === v.mask;
  const isNeighbour =
    state.hoveredMask !== null && isOneBitApart(state.hoveredMask, v.mask);
  const inPath = state.activePath?.includes(v.mask) ?? false;

  const scale = isHover ? 1.18 : isNeighbour ? 1.06 : 1;
  const radius = r * scale;

  if (isHover) {
    ctx.beginPath();
    ctx.arc(v.x, v.y, radius + 7, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, 0.06)`;
    ctx.fill();
  }

  ctx.beginPath();
  ctx.arc(v.x, v.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = divergingColor(v.fitness, state.colorRange);
  ctx.fill();

  const isSelected = state.highlightMask === v.mask;
  const isCursor = state.cursorMask === v.mask;
  if (isCursor || isSelected) {
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = HIGHLIGHT;
  } else if (inPath || isHover) {
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, 0.85)`;
  } else {
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, 0.32)`;
  }
  ctx.stroke();

  if (isCursor) {
    ctx.beginPath();
    ctx.arc(v.x, v.y, radius + 5, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, 0.9)`;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (v.measured !== null) {
    ctx.beginPath();
    ctx.arc(v.x, v.y, Math.max(3, radius * 0.28), 0, Math.PI * 2);
    ctx.fillStyle = rgb(INK);
    ctx.fill();
  }

  if (v.mask === 0) {
    ctx.fillStyle = rgb(INK);
    ctx.font = "600 10px Inter, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("WT", v.x, v.y - radius - 10);
  }
}

function drawVertices(ctx: CanvasRenderingContext2D, state: RenderState) {
  for (const v of state.lattice.vertices.values()) {
    if (state.visibleMasks && !state.visibleMasks.has(v.mask)) continue;
    drawVertex(ctx, v, state);
  }
}

function drawRowLabels(
  ctx: CanvasRenderingContext2D,
  state: RenderState,
  width: number,
) {
  const { lattice } = state;
  ctx.fillStyle = `rgba(${INK[0]}, ${INK[1]}, ${INK[2]}, 0.38)`;
  ctx.font = "500 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let d = 0; d <= lattice.k; d++) {
    const row = lattice.byDepth[d];
    if (row.length === 0) continue;
    const visibleRow = state.visibleMasks
      ? row.filter((v) => state.visibleMasks!.has(v.mask))
      : row;
    if (visibleRow.length === 0) continue;
    const y = visibleRow[0].y;
    ctx.fillText(`d = ${d}`, 12, y);
    ctx.textAlign = "right";
    ctx.fillText(`${visibleRow.length}`, width - 12, y);
    ctx.textAlign = "left";
  }
}

function isOneBitApart(a: number, b: number): boolean {
  const diff = a ^ b;
  return diff !== 0 && (diff & (diff - 1)) === 0;
}

export function render(
  ctx: CanvasRenderingContext2D,
  state: RenderState,
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height);
  drawEdges(ctx, state);
  drawVertices(ctx, state);
  drawRowLabels(ctx, state, width);
}

export function setupCanvasDPR(
  canvas: HTMLCanvasElement,
): { ctx: CanvasRenderingContext2D; width: number; height: number } {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

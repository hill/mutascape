import { parseFasta } from "./fasta";
import {
  SyntheticModel,
  gb1SingleOverrides,
  measuredByMask,
  sitesMatchOlson,
} from "./fitness";
import {
  DENSE_K_LIMIT,
  buildLattice,
  findVertexAt,
  hammingDistance,
  layoutLattice,
  scoreLattice,
} from "./lattice";
import { probeOracle, scoreViaOracle } from "./oracle";
import { render, setupCanvasDPR, type RenderState } from "./render";
import { Rng } from "./rng";
import { AAS, type ChoiceSite, type Gb1Data, type Lattice } from "./types";
import { greedyWalk, localOptima, shortestPathFromWT } from "./walk";

const MIN_K = 2;
const HARD_MAX_K = 20;

interface Mode {
  protein: string;
  wt: string;
  gb1: Gb1Data | null;
}

interface AppState {
  mode: Mode;
  selectedPositions: number[];
  altAAs: Map<number, string>;
  alpha: number;
  seed: number;
  hoveredMask: number | null;
  cursorMask: number;
  activePath: number[] | null;
  lattice: Lattice;
  optima: number[];
  oracleUp: boolean;
  /** Monotonic id so a slow oracle response from an earlier rebuild is
   * ignored when a newer rebuild has already fired. */
  rescoreTicket: number;
}

function el<T extends HTMLElement>(selector: string): T {
  const e = document.querySelector(selector);
  if (!e) throw new Error(`missing element: ${selector}`);
  return e as T;
}

const dom = {
  seqStrip: el<HTMLDivElement>("#sequence-strip"),
  proteinLabel: el<HTMLSpanElement>("#protein-label"),
  positions: el<HTMLDivElement>("#positions"),
  vertexCount: el<HTMLSpanElement>("#vertex-count"),
  focusHint: el<HTMLDivElement>("#focus-hint"),
  alpha: el<HTMLInputElement>("#alpha"),
  alphaValue: el<HTMLSpanElement>("#alpha-value"),
  alphaNote: el<HTMLDivElement>("#alpha-note"),
  greedy: el<HTMLButtonElement>("#greedy"),
  reseed: el<HTMLButtonElement>("#reseed"),
  resetView: el<HTMLButtonElement>("#reset-view"),
  canvas: el<HTMLCanvasElement>("#lattice"),
  tooltip: el<HTMLDivElement>("#tooltip"),
  statOptima: el<HTMLSpanElement>("#stat-optima"),
  statBest: el<HTMLSpanElement>("#stat-best"),
  statBestF: el<HTMLSpanElement>("#stat-best-f"),
  statWtRank: el<HTMLSpanElement>("#stat-wt-rank"),
  loadGb1: el<HTMLButtonElement>("#load-gb1"),
  pasteFasta: el<HTMLButtonElement>("#paste-fasta"),
  oracleBadge: el<HTMLSpanElement>("#oracle-badge"),
  fastaDialog: el<HTMLDialogElement>("#fasta-dialog"),
  fastaInput: el<HTMLTextAreaElement>("#fasta-input"),
  fastaSubmit: el<HTMLButtonElement>("#fasta-submit"),
  fastaError: el<HTMLDivElement>("#fasta-error"),
  toast: el<HTMLDivElement>("#toast"),
  variantPanel: el<HTMLDivElement>("#variant-panel"),
  variantLabel: el<HTMLSpanElement>("#variant-label"),
  variantDepth: el<HTMLSpanElement>("#variant-depth"),
  variantFitness: el<HTMLSpanElement>("#variant-fitness"),
  variantMeasuredWrap: el<HTMLSpanElement>("#variant-measured-wrap"),
  variantMeasured: el<HTMLSpanElement>("#variant-measured"),
  variantStrip: el<HTMLDivElement>("#variant-strip"),
};

const state = {} as AppState;

async function boot() {
  state.oracleUp = false;
  state.rescoreTicket = 0;

  const [gb1, meta] = await Promise.all([
    fetch("./data/gb1.json").then((r) => r.json() as Promise<Gb1Data>),
    probeOracle(),
  ]);

  if (meta?.oracle) {
    state.oracleUp = true;
    dom.oracleBadge.textContent = "oracle live";
    dom.oracleBadge.classList.add("live");
  }

  setGb1Mode(gb1);
  wireEvents();
  rebuild();
}

function setGb1Mode(gb1: Gb1Data) {
  state.mode = { protein: "GB1", wt: gb1.wt, gb1 };
  state.selectedPositions = [...gb1.olson.positions];
  state.altAAs = new Map();
  for (const pos of gb1.olson.positions) {
    const row = gb1.olson.singles[String(pos)];
    state.altAAs.set(pos, bestAltAA(row, gb1.wt[pos - 1]));
  }
  state.alpha = 1.0;
  state.seed = 42;
  state.activePath = null;
  state.hoveredMask = null;
  state.cursorMask = 0;
  dom.alpha.value = "1.0";
  dom.alphaValue.textContent = "1.00";
}

function setCustomMode(wt: string, name: string) {
  state.mode = { protein: name, wt, gb1: null };
  const step = Math.max(1, Math.floor(wt.length / 4));
  const proposed = [1, 1 + step, 1 + 2 * step, Math.min(wt.length, 1 + 3 * step)];
  state.selectedPositions = Array.from(new Set(proposed))
    .filter((p) => p >= 1 && p <= wt.length)
    .sort((a, b) => a - b);

  state.altAAs = new Map();
  const rng = new Rng(42);
  for (const pos of state.selectedPositions) {
    const wtAA = wt[pos - 1];
    let alt = AAS[Math.floor(rng.next() * AAS.length)];
    if (alt === wtAA) alt = AAS[(AAS.indexOf(alt) + 1) % AAS.length];
    state.altAAs.set(pos, alt);
  }
  state.alpha = 1.0;
  state.seed = 42;
  state.activePath = null;
  state.hoveredMask = null;
  state.cursorMask = 0;
  dom.alpha.value = "1.0";
  dom.alphaValue.textContent = "1.00";
}

function bestAltAA(row: Record<string, number> | undefined, wtAA: string): string {
  if (!row) return AAS[0] === wtAA ? AAS[1] : AAS[0];
  let bestAA = "";
  let bestF = -Infinity;
  for (const [aa, f] of Object.entries(row)) {
    if (aa === wtAA) continue;
    if (f > bestF) {
      bestF = f;
      bestAA = aa;
    }
  }
  return bestAA || (AAS[0] === wtAA ? AAS[1] : AAS[0]);
}

function currentSites(): ChoiceSite[] {
  return state.selectedPositions.map((position) => ({
    position,
    wtAA: state.mode.wt[position - 1],
    altAA: state.altAAs.get(position) ?? "A",
  }));
}

function rebuild() {
  const sites = currentSites();
  const rng = new Rng(state.seed);
  const gb1 = state.mode.gb1;
  const overrides = gb1 ? gb1SingleOverrides(gb1, sites) : undefined;
  const model = new SyntheticModel(sites, state.alpha, rng, overrides);

  const lat = buildLattice(sites, { cursorMask: state.cursorMask });
  const measured =
    gb1 && sitesMatchOlson(gb1, sites)
      ? measuredByMask(gb1, sites, lat.vertices.keys())
      : null;
  scoreLattice(lat, (m) => model.score(m), measured);

  state.lattice = lat;
  state.optima = localOptima(lat);

  dom.vertexCount.textContent = `${lat.vertices.size}${lat.isDense ? "" : " of 2^" + lat.k}`;
  renderFocusHint();
  renderSequenceStrip();
  renderPositionsPanel();
  renderStats();
  renderAlphaNote();
  renderVariantPanel();
  doLayoutAndRender();

  if (state.oracleUp) void rescoreFromOracle();
}

async function rescoreFromOracle() {
  const ticket = ++state.rescoreTicket;
  const wt = state.mode.wt;
  const sites = currentSites();
  const masks = Array.from(state.lattice.vertices.keys());
  try {
    const resp = await scoreViaOracle(wt, sites, state.alpha, state.seed, masks);
    if (ticket !== state.rescoreTicket) return;
    for (const { mask, fitness } of resp.scores) {
      const v = state.lattice.vertices.get(mask);
      if (v) v.fitness = fitness;
    }
    state.optima = localOptima(state.lattice);
    renderStats();
    renderVariantPanel();
    doLayoutAndRender();
  } catch {
    // Silent: leave the client-side prediction in place.
  }
}

function renderFocusHint() {
  if (state.lattice.isDense) {
    dom.focusHint.hidden = true;
    return;
  }
  dom.focusHint.hidden = false;
  dom.focusHint.textContent = `focus mode: showing Hamming-${state.lattice.focusRadius} neighbourhood around cursor. click a vertex to recenter.`;
}

function renderSequenceStrip() {
  const wt = state.mode.wt;
  const selected = new Set(state.selectedPositions);
  const frag = document.createDocumentFragment();

  for (let i = 0; i < wt.length; i++) {
    const pos = i + 1;
    const span = document.createElement("span");
    span.className = "residue" + (selected.has(pos) ? " selected" : "");
    span.dataset.position = String(pos);
    span.textContent = wt[i];

    const label = document.createElement("span");
    label.className = "pos-label";
    label.textContent = pos % 10 === 0 || selected.has(pos) ? String(pos) : "";
    span.appendChild(label);
    span.title = `${wt[i]}${pos}`;
    frag.appendChild(span);
  }
  dom.seqStrip.replaceChildren(frag);
  dom.proteinLabel.textContent = `${state.mode.protein} · ${wt.length} aa`;
}

function renderPositionsPanel() {
  const frag = document.createDocumentFragment();
  for (const pos of state.selectedPositions) {
    const row = document.createElement("div");
    row.className = "position-row";
    const wtAA = state.mode.wt[pos - 1];
    const alt = state.altAAs.get(pos) ?? "A";

    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.innerHTML = `<span class="wt-aa">${wtAA}${pos}</span> &rarr; `;

    const select = document.createElement("select");
    for (const aa of AAS) {
      if (aa === wtAA) continue;
      const opt = document.createElement("option");
      opt.value = aa;
      opt.textContent = aa;
      if (aa === alt) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      state.altAAs.set(pos, select.value);
      state.activePath = null;
      state.cursorMask = 0;
      rebuild();
    });

    const wrap = document.createElement("span");
    wrap.appendChild(lbl);
    wrap.appendChild(select);
    row.appendChild(wrap);
    frag.appendChild(row);
  }
  dom.positions.replaceChildren(frag);
}

function renderStats() {
  const lat = state.lattice;
  dom.statOptima.textContent =
    state.optima.length + (lat.isDense ? "" : " (local)");

  let bestV = lat.vertices.get(state.cursorMask) ?? lat.vertices.values().next().value!;
  for (const v of lat.vertices.values()) {
    if (v.fitness > bestV.fitness) bestV = v;
  }
  dom.statBest.textContent = maskLabel(bestV.mask);
  dom.statBestF.textContent = bestV.fitness.toFixed(3);

  const wtVert = lat.vertices.get(0);
  if (wtVert) {
    const wtF = wtVert.fitness;
    let worse = 0;
    for (const v of lat.vertices.values()) if (v.fitness <= wtF) worse++;
    const pct = ((worse / lat.vertices.size) * 100).toFixed(0);
    dom.statWtRank.textContent = `${pct}%`;
  } else {
    dom.statWtRank.textContent = "·";
  }
}

function renderAlphaNote() {
  const sites = currentSites();
  const showsMeasured =
    state.mode.gb1 !== null && sitesMatchOlson(state.mode.gb1, sites);
  dom.alphaNote.textContent = showsMeasured
    ? "α tunes predicted fitness. measured dots (Olson) are fixed."
    : "α scales epistasis (0 = additive, higher = more rugged)";
}

function renderVariantPanel() {
  const lat = state.lattice;
  const cursor = lat.vertices.get(state.cursorMask);
  if (!cursor) {
    dom.variantPanel.hidden = true;
    return;
  }
  dom.variantPanel.hidden = false;

  dom.variantLabel.innerHTML =
    cursor.mask === 0
      ? `<span class="wt-tag">WT &middot;</span> ${state.mode.protein}`
      : maskLabel(cursor.mask);
  dom.variantDepth.textContent = `${cursor.depth}${lat.isDense ? "" : " / " + lat.k}`;

  const fitText = cursor.fitness >= 0 ? `+${cursor.fitness.toFixed(3)}` : cursor.fitness.toFixed(3);
  dom.variantFitness.textContent = fitText;
  dom.variantFitness.className =
    "pill " + (cursor.fitness > 0.001 ? "good" : cursor.fitness < -0.001 ? "bad" : "");

  if (cursor.measured !== null) {
    dom.variantMeasuredWrap.hidden = false;
    const mText =
      cursor.measured >= 0 ? `+${cursor.measured.toFixed(3)}` : cursor.measured.toFixed(3);
    dom.variantMeasured.textContent = mText;
    dom.variantMeasured.className =
      "pill " + (cursor.measured > 0.001 ? "good" : cursor.measured < -0.001 ? "bad" : "");
  } else {
    dom.variantMeasuredWrap.hidden = true;
  }

  renderVariantStrip();
}

/** Render the cursor variant's full sequence, colouring each mutated position
 * by whether the mutation improves fitness relative to WT's predicted baseline. */
function renderVariantStrip() {
  const wt = state.mode.wt;
  const cursor = state.lattice.vertices.get(state.cursorMask)!;
  const sites = state.lattice.sites;
  const wtF = state.lattice.vertices.get(0)?.fitness ?? 0;
  const beneficial = cursor.fitness > wtF;

  const mutated = new Map<number, string>();
  for (let b = 0; b < sites.length; b++) {
    if ((cursor.mask >> b) & 1) {
      mutated.set(sites[b].position, sites[b].altAA);
    }
  }

  const frag = document.createDocumentFragment();
  for (let i = 0; i < wt.length; i++) {
    const pos = i + 1;
    const span = document.createElement("span");
    const alt = mutated.get(pos);
    span.className = "residue" + (alt ? " mutated" + (beneficial ? " beneficial" : "") : "");
    span.dataset.position = String(pos);
    span.textContent = alt ?? wt[i];

    const label = document.createElement("span");
    label.className = "pos-label";
    label.textContent = pos % 10 === 0 || alt ? String(pos) : "";
    span.appendChild(label);
    if (alt) span.title = `${wt[i]}${pos}${alt}`;
    frag.appendChild(span);
  }
  dom.variantStrip.replaceChildren(frag);
}

function maskLabel(mask: number): string {
  if (mask === 0) return "WT";
  const parts: string[] = [];
  const sites = state.lattice.sites;
  for (let b = 0; b < sites.length; b++) {
    if ((mask >> b) & 1) {
      parts.push(`${sites[b].wtAA}${sites[b].position}${sites[b].altAA}`);
    }
  }
  return parts.join(" + ");
}

function colorRange(): number {
  let max = 0.0001;
  for (const v of state.lattice.vertices.values()) {
    if (Math.abs(v.fitness) > max) max = Math.abs(v.fitness);
  }
  return max;
}

function renderState(): RenderState {
  return {
    lattice: state.lattice,
    hoveredMask: state.hoveredMask,
    activePath: state.activePath,
    pathProgress: 1,
    colorRange: colorRange(),
    highlightMask: null,
    visibleMasks: null,
    cursorMask: state.cursorMask,
  };
}

function doLayoutAndRender() {
  const { ctx, width, height } = setupCanvasDPR(dom.canvas);
  layoutLattice(state.lattice, width, height);
  render(ctx, renderState(), width, height);
}

function updateTooltip(mask: number | null, clientX: number, clientY: number) {
  if (mask === null) {
    dom.tooltip.hidden = true;
    return;
  }
  const v = state.lattice.vertices.get(mask);
  if (!v) {
    dom.tooltip.hidden = true;
    return;
  }
  const row = state.lattice.byDepth[v.depth];
  const rank = row.filter((n) => n.fitness > v.fitness).length + 1;
  const optimal = state.optima.includes(mask);

  const lines: string[] = [];
  lines.push(`<div class="tt-label">${maskLabel(mask)}</div>`);
  lines.push(
    `<div class="tt-row">fitness <span>${v.fitness.toFixed(3)}</span></div>`,
  );
  if (v.measured !== null) {
    lines.push(
      `<div class="tt-row">measured <span>${v.measured.toFixed(3)}</span></div>`,
    );
  }
  lines.push(
    `<div class="tt-row">d = ${v.depth} · rank ${rank}/${row.length}${optimal ? " · optimum" : ""}</div>`,
  );

  dom.tooltip.innerHTML = lines.join("");
  dom.tooltip.hidden = false;
  const rect = dom.canvas.parentElement!.getBoundingClientRect();
  dom.tooltip.style.left = `${clientX - rect.left}px`;
  dom.tooltip.style.top = `${clientY - rect.top}px`;
}

function wireEvents() {
  dom.alpha.addEventListener("input", () => {
    state.alpha = Number(dom.alpha.value);
    dom.alphaValue.textContent = state.alpha.toFixed(2);
    rebuild();
  });

  dom.greedy.addEventListener("click", () => {
    animatePath(greedyWalk(state.lattice, 0));
  });

  dom.reseed.addEventListener("click", () => {
    state.seed = Math.floor(Math.random() * 1e9);
    state.activePath = null;
    rebuild();
    toast("reseeded couplings");
  });

  dom.resetView.addEventListener("click", () => {
    state.activePath = null;
    state.alpha = 1.0;
    state.cursorMask = 0;
    dom.alpha.value = "1.0";
    dom.alphaValue.textContent = "1.00";
    rebuild();
  });

  dom.loadGb1.addEventListener("click", async () => {
    const gb1 = await fetch("./data/gb1.json").then((r) => r.json());
    setGb1Mode(gb1);
    rebuild();
    toast("loaded GB1");
  });

  dom.pasteFasta.addEventListener("click", () => {
    dom.fastaError.hidden = true;
    dom.fastaInput.value = "";
    dom.fastaDialog.showModal();
  });
  dom.fastaSubmit.addEventListener("click", (e) => {
    try {
      const rec = parseFasta(dom.fastaInput.value);
      setCustomMode(rec.sequence, rec.header);
      rebuild();
      dom.fastaDialog.close();
      toast(
        rec.extraRecords > 0
          ? `loaded ${rec.header} (${rec.extraRecords} extra records ignored)`
          : `loaded ${rec.header}`,
      );
    } catch (err) {
      e.preventDefault();
      dom.fastaError.textContent = (err as Error).message;
      dom.fastaError.hidden = false;
    }
  });

  dom.seqStrip.addEventListener("click", (e) => {
    const target = (e.target as HTMLElement).closest(".residue") as HTMLElement | null;
    if (!target) return;
    const pos = Number(target.dataset.position);
    toggleSelected(pos);
  });

  dom.canvas.addEventListener("mousemove", (e) => {
    const rect = dom.canvas.getBoundingClientRect();
    const v = findVertexAt(
      state.lattice,
      e.clientX - rect.left,
      e.clientY - rect.top,
      22,
    );
    const newMask = v ? v.mask : null;
    if (newMask !== state.hoveredMask) {
      state.hoveredMask = newMask;
      doLayoutAndRender();
    }
    if (newMask !== null) updateTooltip(newMask, e.clientX, e.clientY);
    else dom.tooltip.hidden = true;
  });
  dom.canvas.addEventListener("mouseleave", () => {
    state.hoveredMask = null;
    dom.tooltip.hidden = true;
    doLayoutAndRender();
  });
  dom.canvas.addEventListener("click", (e) => {
    const rect = dom.canvas.getBoundingClientRect();
    const v = findVertexAt(
      state.lattice,
      e.clientX - rect.left,
      e.clientY - rect.top,
      22,
    );
    if (!v) return;
    selectVariant(v.mask);
  });

  window.addEventListener("resize", () => doLayoutAndRender());
}

function selectVariant(mask: number) {
  const previousCursor = state.cursorMask;
  state.cursorMask = mask;

  if (!state.lattice.isDense && hammingDistance(previousCursor, mask) > 0) {
    // In focus mode, re-centering the cursor recomputes the visible ball
    // around the new cursor. Rebuild first, then animate the path if the
    // whole path is still visible.
    rebuild();
  }

  if (mask !== 0) {
    const path = shortestPathFromWT(state.lattice, mask);
    const allVisible = path.every((m) => state.lattice.vertices.has(m));
    if (allVisible) {
      animatePath(path);
    } else {
      state.activePath = null;
      doLayoutAndRender();
      renderVariantPanel();
    }
  } else {
    state.activePath = null;
    doLayoutAndRender();
    renderVariantPanel();
  }
}

function toggleSelected(pos: number) {
  const s = new Set(state.selectedPositions);
  if (s.has(pos)) {
    if (s.size <= MIN_K) {
      toast(`need at least ${MIN_K} positions`);
      return;
    }
    s.delete(pos);
    state.altAAs.delete(pos);
  } else {
    if (s.size >= HARD_MAX_K) {
      toast(`lattice size limit: 2^${HARD_MAX_K} variants`);
      return;
    }
    s.add(pos);
    const wtAA = state.mode.wt[pos - 1];
    if (state.mode.gb1) {
      const row = state.mode.gb1.olson.singles[String(pos)];
      state.altAAs.set(pos, bestAltAA(row, wtAA));
    } else {
      const rng = new Rng(state.seed + pos);
      let alt = AAS[Math.floor(rng.next() * AAS.length)];
      if (alt === wtAA) alt = AAS[(AAS.indexOf(alt) + 1) % AAS.length];
      state.altAAs.set(pos, alt);
    }
  }
  state.selectedPositions = [...s].sort((a, b) => a - b);
  state.activePath = null;
  state.cursorMask = 0;
  rebuild();
}

function animatePath(path: number[]) {
  if (path.length < 2) {
    state.activePath = path;
    doLayoutAndRender();
    renderVariantPanel();
    return;
  }
  state.activePath = [path[0]];
  doLayoutAndRender();

  let i = 1;
  const step = () => {
    if (i >= path.length) return;
    state.activePath = path.slice(0, i + 1);
    doLayoutAndRender();
    i++;
    setTimeout(step, 220);
  };
  setTimeout(step, 220);
}

let toastTimer: number | undefined;
function toast(msg: string) {
  dom.toast.textContent = msg;
  dom.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, 1800);
}

// DENSE_K_LIMIT is imported only so its narrower semantics are visible to
// readers when they jump to the rebuild path; the actual threshold lives in
// lattice.ts.
void DENSE_K_LIMIT;

boot();

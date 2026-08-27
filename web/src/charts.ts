import type { KumoChartOption } from "@cloudflare/kumo"
import type { SeriesOption } from "echarts"
import type {
  LandscapeGraph,
  PosteriorProfilePoint,
  PosteriorQueryResult,
  ProfilePoint,
  ScoreHistogram,
  SubstitutionMap,
  WalshPreview,
} from "./types"

const REAL = "#0f172a"
const SYNTHETIC = "#27ae60"

// Cells stop being readable below roughly this width, so wider scans get a zoom slider
// rather than more columns squeezed into the same card.
const ZOOM_THRESHOLD = 24
const RAMP_STOPS = 33
export const DIVERGING = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]

function profileMap(points: ProfilePoint[]): Map<number, ProfilePoint> {
  return new Map(points.map((point) => [point.depth, point]))
}

function quantileData(
  depths: number[],
  points: Map<number, ProfilePoint>,
  key: keyof Pick<ProfilePoint, "q05" | "q25" | "median" | "q75" | "q95">,
): Array<number | null> {
  return depths.map((depth) => points.get(depth)?.[key] ?? null)
}

function difference(high: Array<number | null>, low: Array<number | null>): Array<number | null> {
  return high.map((value, index) => {
    const base = low[index]
    return value === null || base === null ? null : value - base
  })
}

function bandSeries(
  label: string,
  colour: string,
  depths: number[],
  points: ProfilePoint[],
): SeriesOption[] {
  const mapped = profileMap(points)
  const q05 = quantileData(depths, mapped, "q05")
  const q25 = quantileData(depths, mapped, "q25")
  const median = quantileData(depths, mapped, "median")
  const q75 = quantileData(depths, mapped, "q75")
  const q95 = quantileData(depths, mapped, "q95")
  const slug = label.toLowerCase()
  return [
    {
      name: `${label} q05`,
      type: "line",
      data: q05,
      stack: `${slug}-outer`,
      stackStrategy: "all",
      symbol: "none",
      lineStyle: { opacity: 0 },
      areaStyle: { opacity: 0 },
      tooltip: { show: false },
      silent: true,
    },
    {
      name: `${label} 5 to 95%`,
      type: "line",
      data: difference(q95, q05),
      stack: `${slug}-outer`,
      stackStrategy: "all",
      symbol: "none",
      lineStyle: { opacity: 0 },
      areaStyle: { color: colour, opacity: 0.08 },
      tooltip: { show: false },
      silent: true,
    },
    {
      name: `${label} q25`,
      type: "line",
      data: q25,
      stack: `${slug}-inner`,
      stackStrategy: "all",
      symbol: "none",
      lineStyle: { opacity: 0 },
      areaStyle: { opacity: 0 },
      tooltip: { show: false },
      silent: true,
    },
    {
      name: `${label} 25 to 75%`,
      type: "line",
      data: difference(q75, q25),
      stack: `${slug}-inner`,
      stackStrategy: "all",
      symbol: "none",
      lineStyle: { opacity: 0 },
      areaStyle: { color: colour, opacity: 0.2 },
      tooltip: { show: false },
      silent: true,
    },
    {
      name: label,
      type: "line",
      data: median,
      symbol: "circle",
      symbolSize: 7,
      lineStyle: { color: colour, width: 2 },
      itemStyle: { color: colour },
    },
  ]
}

export function fitnessProfileOptions(profiles: {
  real?: ProfilePoint[]
  synthetic: ProfilePoint[]
}): KumoChartOption {
  const depths = Array.from(
    new Set([
      ...(profiles.real ?? []).map((point) => point.depth),
      ...profiles.synthetic.map((point) => point.depth),
    ]),
  ).sort((a, b) => a - b)
  const series = [
    ...(profiles.real ? (bandSeries("Measured", REAL, depths, profiles.real) ?? []) : []),
    ...(bandSeries("Synthetic", SYNTHETIC, depths, profiles.synthetic) ?? []),
  ]
  return {
    animationDuration: 250,
    aria: { enabled: true },
    color: [REAL, SYNTHETIC],
    grid: { left: 54, right: 24, top: 44, bottom: 48 },
    legend: { data: profiles.real ? ["Measured", "Synthetic"] : ["Synthetic"] },
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: unknown) =>
        typeof value === "number" ? value.toFixed(3) : String(value),
    },
    xAxis: {
      type: "category",
      data: depths,
      name: "Mutation depth",
      nameLocation: "middle",
      nameGap: 30,
    },
    yAxis: { type: "value", name: "Fitness", scale: true },
    series,
  }
}

export function walshOptions(walsh: WalshPreview): KumoChartOption {
  return {
    animationDuration: 250,
    aria: { enabled: true },
    color: [REAL, SYNTHETIC],
    grid: { left: 54, right: 24, top: 44, bottom: 46 },
    legend: { data: ["Measured", "Synthetic"] },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: walsh.orders.map((order) => `Order ${order}`),
    },
    yAxis: { type: "value", name: "Variance fraction", max: 1 },
    series: [
      {
        name: "Measured",
        type: "bar",
        data: walsh.orders.map((order) => walsh.real[String(order)]),
      },
      {
        name: "Synthetic",
        type: "bar",
        data: walsh.orders.map((order) => walsh.synthetic[String(order)]),
      },
    ],
  }
}

/**
 * The variant itself, every variant it reaches by adding substitutions, and the lineage
 * back to the parent. Both directions matter, since a path is only meaningful when you can
 * see where it started as well as where it goes.
 */
export function reachableFrom(graph: LandscapeGraph, id: string | null): Set<string> | null {
  if (!id || !graph.nodes.some((node) => node.id === id)) return null
  const forward = new Map<string, string[]>()
  const backward = new Map<string, string[]>()
  for (const edge of graph.edges) {
    forward.set(edge.source, [...(forward.get(edge.source) ?? []), edge.target])
    backward.set(edge.target, [...(backward.get(edge.target) ?? []), edge.source])
  }
  const found = new Set<string>([id])
  for (const links of [forward, backward]) {
    const queue = [id]
    while (queue.length > 0) {
      const current = queue.pop() as string
      for (const next of links.get(current) ?? []) {
        if (found.has(next)) continue
        found.add(next)
        queue.push(next)
      }
    }
  }
  return found
}

// Every variant at one depth would otherwise land on a single vertical line and overplot,
// so each node takes a fixed offset within its column. Keyed on the id rather than drawn at
// random, because the panel re-renders on every selection and nodes must not move.
function columnOffset(id: string): number {
  let hash = 2166136261
  for (let index = 0; index < id.length; index++) {
    hash ^= id.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return ((hash >>> 0) / 4294967295 - 0.5) * 0.62
}

export function landscapeGraphOptions(
  graph: LandscapeGraph,
  selectedId: string | null = null,
): KumoChartOption {
  const fitness = graph.nodes.map((node) => node.fitness)
  const lower = Math.min(...fitness, 0)
  const upper = Math.max(...fitness, 0)
  const maximumDepth = Math.max(...graph.nodes.map((node) => node.depth), 1)
  const maximumUncertainty = Math.max(...graph.nodes.map((node) => node.posteriorStd ?? 0), 0)
  const reachable = reachableFrom(graph, selectedId)
  // Naming the strongest few variants is what makes the peaks readable without clicking.
  // Only the two extremes get a standing label. Naming a cluster of near-equal peaks stacks
  // long variant strings on top of each other, and hideOverlap does not apply to graph series.
  const ranked = [...graph.nodes]
    .filter((node) => node.id !== "wt")
    .sort((first, second) => second.fitness - first.fitness)
  const named = new Set(
    [ranked.at(0)?.id, ranked.at(-1)?.id].filter((id): id is string => id !== undefined),
  )
  const data = graph.nodes.map((node) => {
    const inPath = reachable === null || reachable.has(node.id)
    const parent = node.id === "wt"
    const labelled = parent || node.id === selectedId || (inPath && named.has(node.id))
    return {
      id: node.id,
      name: node.variant,
      variant: node.variant,
      fitness: node.fitness,
      depth: node.depth,
      posteriorStd: node.posteriorStd,
      value: [parent ? 0 : node.depth + columnOffset(node.id), node.fitness],
      symbol: parent ? "rect" : "circle",
      symbolSize:
        (parent ? 15 : node.sampled ? 11 : 7) +
        (node.id === selectedId ? 6 : 0) +
        (maximumUncertainty > 0 ? ((node.posteriorStd ?? 0) / maximumUncertainty) * 6 : 0),
      label: {
        show: labelled,
        formatter: parent ? "Parent" : node.variant,
        position: parent ? ("bottom" as const) : ("top" as const),
        fontSize: 10,
        color: "#3f3f46",
      },
      itemStyle: {
        opacity: inPath ? (node.sampled || parent ? 0.95 : 0.55) : 0.1,
        borderColor: node.id === selectedId ? "#0f172a" : "#ffffff",
        borderWidth: node.id === selectedId ? 3 : 1,
      },
    }
  })
  const links = graph.edges.map((edge) => ({
    ...edge,
    lineStyle: {
      opacity:
        reachable === null || (reachable.has(edge.source) && reachable.has(edge.target))
          ? 0.28
          : 0.03,
    },
  }))
  return {
    animationDurationUpdate: 180,
    aria: { enabled: true },
    grid: { left: 56, right: 74, top: 18, bottom: 46 },
    tooltip: {
      trigger: "item",
      // Variant labels carry residues read out of an uploaded FASTA, so they are escaped
      // before reaching the tooltip HTML.
      dangerousHtmlFormatter: (params: unknown) => {
        const node = (params as { data?: Record<string, unknown> }).data
        if (!node || typeof node.variant !== "string") return ""
        const depth = node.depth === 0 ? "parent" : `${node.depth} mutation(s)`
        const label = node.id === "wt" ? "Parent" : escapeHtml(node.variant)
        const spread =
          typeof node.posteriorStd === "number" ? `, spread ${node.posteriorStd.toFixed(3)}` : ""
        return `${label}<br>${depth}, fitness ${(node.fitness as number).toFixed(3)}${spread}`
      },
    },
    xAxis: {
      type: "value",
      name: "Mutations from parent",
      nameLocation: "middle",
      nameGap: 26,
      // Whole units either side, because ECharts steps ticks from the minimum. A fractional
      // pad would put every label on a half mutation.
      min: -1,
      max: maximumDepth + 1,
      interval: 1,
      // The dashed parent line is the only reference the plot needs at zero.
      axisLine: { onZero: false },
      axisLabel: {
        formatter: (value: number) =>
          value < 0 || value > maximumDepth ? "" : value === 0 ? "Parent" : String(value),
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      name: "Fitness",
      nameLocation: "middle",
      nameGap: 38,
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: "#f1f1f3" } },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none" },
      { type: "inside", yAxisIndex: 0, filterMode: "none" },
    ],
    visualMap: {
      type: "continuous",
      min: lower,
      max: upper,
      dimension: 1,
      seriesIndex: 0,
      show: false,
      inRange: { color: divergingStops(Math.min(lower, -1e-6), Math.max(upper, 1e-6)) },
    },
    series: [
      {
        type: "graph",
        coordinateSystem: "cartesian2d",
        cursor: "pointer",
        data,
        links,
        labelLayout: { hideOverlap: true },
        lineStyle: { color: "source", width: 1, curveness: 0.06 },
        markLine: {
          silent: true,
          symbol: "none",
          label: {
            formatter: "Parent fitness",
            position: "insideEndTop",
            fontSize: 10,
            color: "#a1a1aa",
          },
          lineStyle: { color: "#d4d4d8", type: "dashed" },
          data: [{ yAxis: 0 }],
        },
        emphasis: {
          focus: "adjacency",
          label: { show: true, position: "top" },
          lineStyle: { opacity: 0.8, width: 2 },
        },
      },
    ],
  }
}

export function posteriorProfileOptions(points: PosteriorProfilePoint[]): KumoChartOption {
  const depths = points.map((point) => point.depth)
  return {
    animationDuration: 250,
    aria: { enabled: true },
    color: [SYNTHETIC, "#7c3aed", "#64748b"],
    grid: { left: 54, right: 24, top: 44, bottom: 48 },
    legend: { data: ["Ensemble median", "Fixed oracle", "Additive"] },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: depths,
      name: "Mutation depth",
      nameLocation: "middle",
      nameGap: 30,
    },
    yAxis: { type: "value", name: "Fitness", scale: true },
    series: [
      ...bandSeries("Ensemble median", SYNTHETIC, depths, points),
      {
        name: "Fixed oracle",
        type: "line",
        data: points.map((point) => point.fixedMedian),
        symbol: "diamond",
        lineStyle: { color: "#7c3aed", width: 2 },
        itemStyle: { color: "#7c3aed" },
      },
      {
        name: "Additive",
        type: "line",
        data: points.map((point) => point.additiveMedian),
        symbol: "none",
        lineStyle: { color: "#64748b", type: "dashed", width: 2 },
      },
    ],
  }
}

export function posteriorQueryOptions(results: PosteriorQueryResult[]): KumoChartOption {
  const lower = results.map((result) => result.intervals["80"].lower)
  const upper = results.map((result) => result.intervals["80"].upper)
  return {
    animationDuration: 250,
    aria: { enabled: true },
    grid: { left: 58, right: 24, top: 44, bottom: 64 },
    legend: { data: ["Ensemble mean", "Fixed oracle", "Additive"] },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: results.map((result) => result.mutationLabel),
      axisLabel: { rotate: results.length > 4 ? 30 : 0 },
    },
    yAxis: { type: "value", name: "Fitness", scale: true },
    series: [
      {
        name: "80% lower",
        type: "line",
        data: lower,
        stack: "posterior-interval",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        silent: true,
      },
      {
        name: "80% interval",
        type: "line",
        data: difference(upper, lower),
        stack: "posterior-interval",
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: SYNTHETIC, opacity: 0.16 },
        silent: true,
      },
      {
        name: "Ensemble mean",
        type: "line",
        data: results.map((result) => result.posteriorMean),
        symbol: "circle",
        lineStyle: { color: SYNTHETIC, width: 2 },
      },
      {
        name: "Fixed oracle",
        type: "line",
        data: results.map((result) => result.fixedOracleFitness),
        symbol: "diamond",
        lineStyle: { color: "#7c3aed", width: 2 },
      },
      {
        name: "Additive",
        type: "line",
        data: results.map((result) => result.additiveFitness),
        symbol: "none",
        lineStyle: { color: "#64748b", type: "dashed", width: 2 },
      },
    ],
  }
}

// A filled "×", drawn over imputed cells so colour stays free to carry the score.
const CROSS =
  "path://M2.6,0 L5,2.4 L7.4,0 L10,2.6 L7.6,5 L10,7.4 L7.4,10 L5,7.6 L2.6,10 L0,7.4 L2.4,5 L0,2.6 Z"

const HTML_ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (character) => HTML_ESCAPES[character])
}

function mixHex(from: string, to: string, t: number): string {
  const parse = (hex: string) =>
    [1, 3, 5].map((start) => Number.parseInt(hex.slice(start, start + 2), 16))
  const [a, b] = [parse(from), parse(to)]
  const channel = (index: number) =>
    Math.round(a[index] + (b[index] - a[index]) * t)
      .toString(16)
      .padStart(2, "0")
  return `#${channel(0)}${channel(1)}${channel(2)}`
}

function rampColour(t: number): string {
  const scaled = Math.min(Math.max(t, 0), 1) * (DIVERGING.length - 1)
  const index = Math.min(Math.floor(scaled), DIVERGING.length - 2)
  return mixHex(DIVERGING[index], DIVERGING[index + 1], scaled - index)
}

/**
 * Colour stops spread evenly over [lower, upper] that still land the neutral colour on
 * zero. ECharts spaces `inRange.color` evenly, so an asymmetric scan given the five raw
 * stops would put white wherever the midpoint of the data fell. Sampling the ramp per
 * arm keeps wild type neutral and gives the shorter arm its full range of colour.
 */
function divergingStops(lower: number, upper: number): string[] {
  return Array.from({ length: RAMP_STOPS }, (_, index) => {
    const value = lower + ((upper - lower) * index) / (RAMP_STOPS - 1)
    return rampColour(0.5 + 0.5 * (value < 0 ? value / -lower : value / upper))
  })
}

export function substitutionMapOptions(map: SubstitutionMap): KumoChartOption {
  const aminoAcids = map.aminoAcids.split("")
  const columns = map.positions.map(
    (position, index) => `${map.wildTypeCompact[index] ?? ""}${position}`,
  )
  // The fourth entry tags provenance for the tooltip, which only receives the value array.
  const measured = map.measured.map((cell) => [...cell, 0])
  const imputed = map.imputed.map((cell) => [...cell, 1])
  const zoomed = map.positions.length > ZOOM_THRESHOLD
  return {
    animationDuration: 250,
    aria: { enabled: true },
    grid: {
      left: 40,
      right: 20,
      top: zoomed ? 48 : 16,
      bottom: zoomed ? 84 : 52,
    },
    tooltip: {
      trigger: "item",
      // `valueFormatter` is handed one number at a time, which cannot name the cell, so
      // the whole item is formatted here instead. The label carries residues read out of
      // an uploaded FASTA, so it is escaped before it reaches the tooltip HTML.
      dangerousHtmlFormatter: (params: unknown) => {
        const value = (params as { value?: unknown }).value
        if (!Array.isArray(value)) return escapeHtml(String(value))
        const [column, row, fitness, imputedFlag] = value as number[]
        const substitution = escapeHtml(`${columns[column ?? 0]}${aminoAcids[row ?? 0]}`)
        if (fitness === undefined) return `${substitution}, parent residue`
        const provenance = imputedFlag === 1 ? "imputed by position mean" : "measured"
        return `${substitution}, ${fitness.toFixed(3)}, ${provenance}`
      },
    },
    visualMap: {
      type: "continuous",
      min: map.lower,
      max: map.upper,
      // Cells carry a provenance flag past the fitness, and the default dimension is the
      // last one, so the colour has to be pinned to fitness explicitly.
      dimension: 2,
      seriesIndex: [0, 1],
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      itemWidth: 12,
      itemHeight: 110,
      text: [`+${map.upper.toFixed(2)}`, map.lower.toFixed(2)],
      inRange: { color: divergingStops(map.lower, map.upper) },
    },
    ...(zoomed
      ? {
          dataZoom: [
            {
              type: "slider",
              xAxisIndex: 0,
              startValue: 0,
              endValue: ZOOM_THRESHOLD - 1,
              top: 4,
              height: 24,
            },
            { type: "inside", xAxisIndex: 0, startValue: 0, endValue: ZOOM_THRESHOLD - 1 },
          ],
        }
      : {}),
    xAxis: {
      type: "category",
      data: columns,
      splitArea: { show: true },
      axisLabel: { interval: 0, rotate: columns.length > 12 ? 60 : 0, fontSize: 10 },
      axisTick: { show: false },
    },
    yAxis: {
      type: "category",
      data: aminoAcids,
      splitArea: { show: true },
      axisLabel: { interval: 0, fontSize: 10 },
      axisTick: { show: false },
    },
    series: [
      {
        name: "Measured",
        type: "heatmap",
        data: measured,
        itemStyle: { borderColor: "#fff", borderWidth: 0.5 },
        progressive: 4000,
      },
      {
        name: "Imputed",
        type: "heatmap",
        data: imputed,
        itemStyle: { borderColor: "#fff", borderWidth: 0.5 },
        progressive: 4000,
      },
      {
        // A cartesian heatmap draws nothing without a visualMap, so the parent residues
        // are marked by a scatter series that carries its own colour.
        name: "Parent",
        type: "scatter",
        data: map.wildTypeCells,
        symbolSize: 5,
        itemStyle: { color: REAL },
      },
      {
        name: "Imputed marker",
        type: "scatter",
        data: imputed,
        symbol: CROSS,
        symbolSize: 5,
        silent: true,
        itemStyle: {
          // The cross has to read against both ends of a diverging ramp, so it flips to
          // white once the cell behind it is saturated enough to swallow a dark mark.
          color: (params: { value: unknown }) => {
            const fitness = Array.isArray(params.value) ? Number(params.value[2]) : 0
            const saturation = fitness < 0 ? fitness / map.lower : fitness / map.upper
            return saturation > 0.45 ? "#ffffff" : "#334155"
          },
          opacity: 0.75,
        },
      },
    ],
  }
}

export function scoreHistogramOptions(histogram: ScoreHistogram): KumoChartOption {
  const labels = histogram.centres.map((centre) => centre.toFixed(2))
  const wildTypeIndex = histogram.centres.reduce(
    (best, centre, index) =>
      Math.abs(centre - histogram.wildType) < Math.abs(histogram.centres[best] - histogram.wildType)
        ? index
        : best,
    0,
  )
  const stacked = histogram.higherOrderCount > 0
  return {
    animationDuration: 250,
    aria: { enabled: true },
    grid: { left: 52, right: 18, top: stacked ? 34 : 18, bottom: 46 },
    ...(stacked ? { legend: { top: 0, itemHeight: 8, itemWidth: 12 } } : {}),
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: labels,
      name: "Fitness relative to parent",
      nameLocation: "middle",
      nameGap: 28,
      axisLabel: { interval: Math.max(0, Math.floor(labels.length / 8)), fontSize: 10 },
    },
    yAxis: { type: "value", name: "Variants", nameGap: 34, nameLocation: "middle" },
    series: [
      {
        name: "Single mutants",
        type: "bar",
        stack: "scores",
        data: histogram.singles,
        barCategoryGap: "10%",
        itemStyle: { color: SYNTHETIC },
        markLine: {
          silent: true,
          symbol: "none",
          label: { formatter: "Parent", position: "insideEndTop", fontSize: 10 },
          lineStyle: { color: REAL, type: "dashed", width: 1.5 },
          data: [{ xAxis: wildTypeIndex }],
        },
      },
      ...(stacked
        ? [
            {
              name: "Combinations",
              type: "bar" as const,
              stack: "scores",
              data: histogram.higherOrder,
              itemStyle: { color: REAL },
            },
          ]
        : []),
    ],
  }
}

export function posteriorDistributionOptions(result: PosteriorQueryResult): KumoChartOption {
  const minimum = Math.min(...result.samples)
  const maximum = Math.max(...result.samples)
  const binCount = 16
  const width = (maximum - minimum) / binCount || 1
  const counts = Array.from({ length: binCount }, () => 0)
  for (const value of result.samples) {
    const index = Math.min(binCount - 1, Math.floor((value - minimum) / width))
    counts[index] += 1
  }
  const centres = counts.map((_, index) => minimum + (index + 0.5) * width)
  return {
    animationDuration: 250,
    aria: { enabled: true },
    grid: { left: 48, right: 18, top: 18, bottom: 52 },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: centres.map((value) => value.toFixed(2)),
      name: "Sampled fitness",
      nameLocation: "middle",
      nameGap: 34,
    },
    yAxis: { type: "value", name: "Draws", minInterval: 1 },
    series: [
      {
        name: "Oracle draws",
        type: "bar",
        data: counts,
        itemStyle: { color: SYNTHETIC },
      },
    ],
  }
}

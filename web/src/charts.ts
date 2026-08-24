import type { KumoChartOption } from "@cloudflare/kumo"
import type { SeriesOption } from "echarts"
import type {
  BenchmarkValidity,
  LandscapeGraph,
  PosteriorProfilePoint,
  PosteriorQueryResult,
  ProfilePoint,
  WalshPreview,
} from "./types"

const REAL = "#0f172a"
const SYNTHETIC = "#f6821f"

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

export function landscapeGraphOptions(graph: LandscapeGraph): KumoChartOption {
  const fitness = graph.nodes.map((node) => node.fitness)
  const minimum = Math.min(...fitness)
  const maximum = Math.max(...fitness)
  const range = maximum - minimum || 1
  const maximumUncertainty = Math.max(...graph.nodes.map((node) => node.posteriorStd ?? 0), 0)
  const data = graph.nodes.map((node) => ({
    id: node.id,
    name: node.variant,
    variant: node.variant,
    fitness: node.fitness,
    depth: node.depth,
    value: [node.depth, node.fitness],
    x: node.depth * 190,
    y: ((maximum - node.fitness) / range) * 430,
    symbolSize:
      (node.id === "wt" ? 15 : node.sampled ? 10 : 6) +
      (maximumUncertainty > 0 ? ((node.posteriorStd ?? 0) / maximumUncertainty) * 8 : 0),
    label: { show: node.id === "wt", formatter: node.id === "wt" ? "WT" : undefined },
    itemStyle: {
      opacity: node.sampled || node.id === "wt" ? 0.95 : 0.58,
      borderColor: node.posteriorStd !== undefined ? "#7c3aed" : "transparent",
      borderWidth:
        maximumUncertainty > 0 ? 1 + ((node.posteriorStd ?? 0) / maximumUncertainty) * 3 : 0,
    },
  }))
  return {
    animationDurationUpdate: 180,
    aria: { enabled: true },
    tooltip: {
      trigger: "item",
      valueFormatter: (value: unknown) =>
        Array.isArray(value) && typeof value[1] === "number"
          ? `depth ${value[0]}, fitness ${value[1].toFixed(3)}`
          : String(value),
    },
    visualMap: {
      type: "continuous",
      min: minimum,
      max: maximum,
      dimension: 1,
      seriesIndex: 0,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      calculable: false,
      text: ["Higher fitness", "Lower fitness"],
      inRange: { color: ["#334155", "#f59e0b", "#ef4444"] },
    },
    series: [
      {
        type: "graph",
        layout: "none",
        roam: true,
        scaleLimit: { min: 0.55, max: 4 },
        data,
        links: graph.edges,
        lineStyle: { color: "source", opacity: 0.22, width: 1 },
        emphasis: {
          focus: "adjacency",
          label: { show: true, position: "right" },
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

export function benchmarkValidityOptions(result: BenchmarkValidity): KumoChartOption {
  return {
    animationDuration: 350,
    aria: { enabled: true },
    color: [SYNTHETIC, "#2563eb", "#10b981"],
    grid: { left: 58, right: 24, top: 54, bottom: 58 },
    legend: { data: ["Rank transfer", "Pairwise winners", "Reliability pass rate"] },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (value: unknown) =>
        typeof value === "number" ? value.toFixed(3) : String(value),
    },
    xAxis: {
      type: "category",
      data: result.methods.map((method) => method.label),
      axisLabel: { interval: 0 },
    },
    yAxis: { type: "value", name: "Score", min: 0, max: 1 },
    series: [
      {
        name: "Rank transfer",
        type: "bar",
        data: result.methods.map((method) => method.kendallTau),
        markLine: {
          symbol: "none",
          label: { formatter: "Measured reliability floor" },
          lineStyle: { color: "#64748b", type: "dashed" },
          data: [{ yAxis: result.measuredReliabilityFloor }],
        },
      },
      {
        name: "Pairwise winners",
        type: "bar",
        data: result.methods.map((method) => method.pairwiseAccuracy),
      },
      {
        name: "Reliability pass rate",
        type: "bar",
        data: result.methods.map((method) => method.reliabilityPassRate),
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

import { describe, expect, test } from "bun:test"
import {
  benchmarkBudgetOptions,
  fitnessProfileOptions,
  landscapeGraphOptions,
  posteriorDistributionOptions,
  posteriorProfileOptions,
  posteriorQueryOptions,
  walshOptions,
} from "./charts"
import type {
  BenchmarkValidity,
  PosteriorProfilePoint,
  PosteriorQueryResult,
  ProfilePoint,
} from "./types"

const point = (depth: number, median: number): ProfilePoint => ({
  depth,
  count: 10,
  q05: median - 2,
  q25: median - 1,
  median,
  q75: median + 1,
  q95: median + 2,
})

describe("fitnessProfileOptions", () => {
  test("aligns measured and synthetic depths", () => {
    const options = fitnessProfileOptions({
      real: [point(1, 0.1), point(2, -0.2)],
      synthetic: [point(1, 0.2), point(3, -0.4)],
    })
    const xAxis = options.xAxis as { data: number[] }
    const series = options.series as Array<{ name: string; data: Array<number | null> }>

    expect(xAxis.data).toEqual([1, 2, 3])
    expect(series.find((item) => item.name === "Measured")?.data).toEqual([0.1, -0.2, null])
    expect(series.find((item) => item.name === "Synthetic")?.data).toEqual([0.2, null, -0.4])
  })

  test("omits measured series for custom previews", () => {
    const options = fitnessProfileOptions({ synthetic: [point(1, 0.2)] })
    const series = options.series as Array<{ name: string }>

    expect(series.some((item) => item.name === "Measured")).toBeFalse()
    expect(series.some((item) => item.name === "Synthetic")).toBeTrue()
  })
})

test("walshOptions keeps interaction order", () => {
  const options = walshOptions({
    orders: [1, 2, 3, 4],
    real: { "1": 0.2, "2": 0.4, "3": 0.3, "4": 0.1 },
    synthetic: { "1": 0.3, "2": 0.4, "3": 0.2, "4": 0.1 },
  })
  const xAxis = options.xAxis as { data: string[] }

  expect(xAxis.data).toEqual(["Order 1", "Order 2", "Order 3", "Order 4"])
})

test("benchmarkBudgetOptions keeps matched development and external series", () => {
  const result = {
    id: "benchmark-validity-v1",
    panel: "development",
    nLandscapes: 16,
    nMethods: 12,
    budget: 96,
    trials: 10,
    measuredReliabilityFloor: 0.25,
    external: {
      panel: "external",
      nLandscapes: 7,
      nMethods: 12,
      budget: 96,
      trials: 10,
      kendallTau: 0.64,
      pairwiseAccuracy: 0.74,
      reliabilityPassRate: 0.57,
    },
    methods: [],
    budgetCurve: {
      budgets: [48, 96, 192, 384],
      series: [
        {
          id: "dms_pairwise_mean",
          label: "Mutascape",
          developmentKendall: [0.53, 0.45, 0.51, 0.46],
          externalKendall: [0.44, 0.64, 0.66, 1],
          developmentWinners: [0.89, 0.79, 0.74, 0.64],
          externalWinners: [0.86, 0.74, 0.85, 0.64],
        },
      ],
    },
  } satisfies BenchmarkValidity

  const options = benchmarkBudgetOptions(result)
  const xAxis = options.xAxis as { data: number[] }
  const series = options.series as Array<{
    name: string
    data: number[]
    lineStyle: { type?: string }
  }>

  expect(xAxis.data).toEqual([48, 96, 192, 384])
  expect(series.map((item) => item.name)).toEqual(["Mutascape, development", "Mutascape, external"])
  expect(series[0].data).toEqual([0.89, 0.79, 0.74, 0.64])
  expect(series[1].lineStyle.type).toBe("dashed")
})

test("landscapeGraphOptions makes the mutation paths interactive", () => {
  const options = landscapeGraphOptions({
    nodes: [
      { id: "wt", variant: "wt", depth: 0, fitness: 0, sampled: false },
      { id: "V39A", variant: "V39A", depth: 1, fitness: 0.4, sampled: true },
    ],
    edges: [{ source: "wt", target: "V39A" }],
  })
  const series = options.series as Array<{
    type: string
    roam: boolean
    data: Array<{ name: string }>
    links: Array<{ source: string; target: string }>
  }>

  expect(series[0].type).toBe("graph")
  expect(series[0].roam).toBeTrue()
  expect(series[0].data.map((node) => node.name)).toEqual(["wt", "V39A"])
  expect(series[0].links).toEqual([{ source: "wt", target: "V39A" }])
})

const posteriorResult: PosteriorQueryResult = {
  index: 0,
  mutationLabel: "V39A+D40G",
  depth: 2,
  observed: false,
  additiveFitness: 0.2,
  fixedOracleFitness: 0.7,
  posteriorMean: 0.5,
  posteriorStd: 0.3,
  intervals: {
    "50": { lower: 0.3, upper: 0.7 },
    "80": { lower: 0.1, upper: 0.9 },
    "95": { lower: -0.2, upper: 1.2 },
  },
  samples: [0.1, 0.3, 0.5, 0.7, 0.9],
}

test("posterior charts expose intervals, oracle draws, and distributions", () => {
  const profile: PosteriorProfilePoint = {
    ...point(2, 0.5),
    fixedMedian: 0.7,
    additiveMedian: 0.2,
    meanStd: 0.3,
  }
  const profileSeries = posteriorProfileOptions([profile]).series as Array<{ name: string }>
  const querySeries = posteriorQueryOptions([posteriorResult]).series as Array<{ name: string }>
  const distributionSeries = posteriorDistributionOptions(posteriorResult).series as Array<{
    data: number[]
  }>

  expect(profileSeries.some((series) => series.name === "Fixed oracle")).toBeTrue()
  expect(querySeries.some((series) => series.name === "80% interval")).toBeTrue()
  expect(distributionSeries[0].data.reduce((sum, value) => sum + value, 0)).toBe(5)
})

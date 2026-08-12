import { describe, expect, test } from "bun:test";
import { fitnessProfileOptions, landscapeGraphOptions, walshOptions } from "./charts";
import type { ProfilePoint } from "./types";

const point = (depth: number, median: number): ProfilePoint => ({
  depth,
  count: 10,
  q05: median - 2,
  q25: median - 1,
  median,
  q75: median + 1,
  q95: median + 2,
});

describe("fitnessProfileOptions", () => {
  test("aligns measured and synthetic depths", () => {
    const options = fitnessProfileOptions({
      real: [point(1, 0.1), point(2, -0.2)],
      synthetic: [point(1, 0.2), point(3, -0.4)],
    });
    const xAxis = options.xAxis as { data: number[] };
    const series = options.series as Array<{ name: string; data: Array<number | null> }>;

    expect(xAxis.data).toEqual([1, 2, 3]);
    expect(series.find((item) => item.name === "Measured")?.data).toEqual([0.1, -0.2, null]);
    expect(series.find((item) => item.name === "Synthetic")?.data).toEqual([0.2, null, -0.4]);
  });

  test("omits measured series for custom previews", () => {
    const options = fitnessProfileOptions({ synthetic: [point(1, 0.2)] });
    const series = options.series as Array<{ name: string }>;

    expect(series.some((item) => item.name === "Measured")).toBeFalse();
    expect(series.some((item) => item.name === "Synthetic")).toBeTrue();
  });
});

test("walshOptions keeps interaction order", () => {
  const options = walshOptions({
    orders: [1, 2, 3, 4],
    real: { "1": 0.2, "2": 0.4, "3": 0.3, "4": 0.1 },
    synthetic: { "1": 0.3, "2": 0.4, "3": 0.2, "4": 0.1 },
  });
  const xAxis = options.xAxis as { data: string[] };

  expect(xAxis.data).toEqual(["Order 1", "Order 2", "Order 3", "Order 4"]);
});

test("landscapeGraphOptions makes the mutation paths interactive", () => {
  const options = landscapeGraphOptions({
    nodes: [
      { id: "wt", variant: "wt", depth: 0, fitness: 0, sampled: false },
      { id: "V39A", variant: "V39A", depth: 1, fitness: 0.4, sampled: true },
    ],
    edges: [{ source: "wt", target: "V39A" }],
  });
  const series = options.series as Array<{
    type: string;
    roam: boolean;
    data: Array<{ name: string }>;
    links: Array<{ source: string; target: string }>;
  }>;

  expect(series[0].type).toBe("graph");
  expect(series[0].roam).toBeTrue();
  expect(series[0].data.map((node) => node.name)).toEqual(["wt", "V39A"]);
  expect(series[0].links).toEqual([{ source: "wt", target: "V39A" }]);
});

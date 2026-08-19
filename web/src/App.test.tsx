import { afterEach, describe, expect, test } from "bun:test"
import { TooltipProvider } from "@cloudflare/kumo"
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router"
import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { createAppRouter } from "./router"

const originalFetch = globalThis.fetch

afterEach(() => {
  cleanup()
  globalThis.fetch = originalFetch
})

describe("landscape builder", () => {
  test("uses routes for stable page navigation", async () => {
    globalThis.fetch = (() => new Promise(() => {})) as unknown as typeof fetch
    const router = createAppRouter(createMemoryHistory({ initialEntries: ["/"] }))
    render(
      <TooltipProvider delay={0} closeDelay={0}>
        <RouterProvider router={router} />
      </TooltipProvider>,
    )

    expect(
      await screen.findByRole("heading", {
        name: "Create landscape",
      }),
    ).toBeTruthy()
    expect(screen.getByRole("img", { name: "mutascape" }).getAttribute("src")).toBe(
      "/assets/logo.png",
    )
    expect(screen.getAllByRole("button", { name: "Collapse sidebar" })).toHaveLength(1)
    expect(screen.getByRole("tab", { name: "Preset data" })).toBeTruthy()

    await userEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }))
    expect(
      screen
        .getByRole("button", { name: "Landscape library" })
        .hasAttribute("data-base-ui-tooltip-trigger"),
    ).toBe(true)

    await userEvent.click(screen.getByRole("tab", { name: "Upload DMS" }))
    expect(screen.getByLabelText(/DMS CSV/)).toBeTruthy()
    expect(screen.getByLabelText(/WT FASTA/)).toBeTruthy()
    expect(
      (screen.getByRole("button", { name: "Use uploaded data" }) as HTMLButtonElement).disabled,
    ).toBe(true)

    await userEvent.click(screen.getByRole("button", { name: "Landscape library" }))

    expect(await screen.findByRole("heading", { name: "Landscape library" })).toBeTruthy()
    expect(router.state.location.pathname).toBe("/landscapes")

    await userEvent.click(screen.getByRole("button", { name: "New landscape" }))
    expect(await screen.findByRole("heading", { name: "Create landscape" })).toBeTruthy()
    expect(router.state.location.pathname).toBe("/")
  })

  test("loads a landscape explorer at its own URL", async () => {
    const model = {
      id: "gb1_headline",
      name: "GB1 reference",
      kind: "reference",
      description: "Reference landscape",
      positions: [39, 40, 41, 54],
      wtCompact: "VDGV",
      parameters: {
        noiseSigma: 0.3,
        alpha: 1.5,
        pairwiseSigma: 0,
        combinatorialUpside: 3,
        seed: 42,
      },
      createdAt: null,
    }
    globalThis.fetch = (async (input) => {
      const url = String(input)
      const body =
        url === "/api/v1/models"
          ? { models: [model] }
          : url === "/api/v1/evaluations"
            ? { results: [{ index: 0, mutationLabel: "V39A+D40G", fitness: 0.731 }] }
            : model
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    }) as typeof fetch
    const router = createAppRouter(
      createMemoryHistory({ initialEntries: ["/landscapes/gb1_headline"] }),
    )
    render(<RouterProvider router={router} />)

    expect(await screen.findByRole("heading", { name: "Explore GB1 reference" })).toBeTruthy()
    expect(screen.getByRole("heading", { name: "Query an exact variant" })).toBeTruthy()
    expect(screen.getByRole("heading", { name: "Sample the landscape" })).toBeTruthy()
    expect(router.state.location.pathname).toBe("/landscapes/gb1_headline")

    await userEvent.type(screen.getByRole("textbox", { name: "Mutation set" }), "V39A+D40G")
    await userEvent.click(screen.getByRole("button", { name: "Score variant" }))
    expect(await screen.findByText("0.7310")).toBeTruthy()
  })

  test("shows posterior uncertainty for an exact query", async () => {
    const model = {
      id: "gb1_posterior",
      name: "GB1 posterior",
      kind: "posterior",
      description: "Posterior landscape",
      positions: [39, 40, 41, 54],
      wtCompact: "VDGV",
      parameters: {
        noiseSigma: 0.3,
        alpha: 1.5,
        pairwiseSigma: 0,
        combinatorialUpside: 3,
        seed: 42,
      },
      posteriorSettings: { priorId: "combinatorial-v2", oracleSeed: 42 },
      createdAt: null,
    }
    globalThis.fetch = (async (input) => {
      const url = String(input)
      const body =
        url === "/api/v1/models"
          ? { models: [model] }
          : url === "/api/v1/posterior-queries"
            ? {
                source: model,
                prior: {
                  id: "combinatorial-v2",
                  name: "Combinatorial v2",
                  description: "Prior",
                  created: "2026-08-12",
                  maxInteractionOrder: 2,
                  nLandscapes: 30,
                  nProteins: 14,
                  sources: [],
                },
                settings: { seed: 42, ensembleSize: 100 },
                results: [
                  {
                    index: 0,
                    mutationLabel: "V39A+D40G",
                    depth: 2,
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
                  },
                ],
              }
            : model
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    }) as typeof fetch
    const router = createAppRouter(
      createMemoryHistory({ initialEntries: ["/landscapes/gb1_posterior"] }),
    )
    render(<RouterProvider router={router} />)

    await userEvent.type(await screen.findByRole("textbox", { name: "Mutation set" }), "V39A+D40G")
    await userEvent.click(screen.getByRole("button", { name: "Score variant" }))

    expect(await screen.findByText("Posterior SD")).toBeTruthy()
    expect(screen.getByText("0.3000")).toBeTruthy()
    expect(screen.getByText("Latest posterior distribution")).toBeTruthy()
  })

  test("shows stored leaderboard results", async () => {
    globalThis.fetch = (async (input) => {
      const url = String(input)
      const body =
        url === "/api/v1/models"
          ? { models: [] }
          : url === "/api/v1/leaderboards"
            ? {
                leaderboards: [
                  {
                    id: "lb_one",
                    name: "GB1 search",
                    landscapeId: "gb1_headline",
                    queryBudget: 1000,
                    createdAt: "2026-08-12T00:00:00Z",
                    entries: [
                      {
                        id: "run_one",
                        modelName: "My optimiser",
                        successes: 8,
                        trials: 10,
                        successRate: 0.8,
                        confidenceLow: 0.49,
                        confidenceHigh: 0.94,
                        createdAt: "2026-08-12T00:00:00Z",
                      },
                    ],
                  },
                ],
              }
            : {}
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    }) as typeof fetch
    const router = createAppRouter(createMemoryHistory({ initialEntries: ["/leaderboards"] }))
    render(<RouterProvider router={router} />)

    expect(await screen.findByRole("heading", { name: "Leaderboards" })).toBeTruthy()
    expect(await screen.findByText("My optimiser")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Create leaderboard" })).toBeTruthy()
  })
})

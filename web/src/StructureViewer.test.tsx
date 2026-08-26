import { afterEach, describe, expect, test } from "bun:test"
import { TooltipProvider } from "@cloudflare/kumo"
import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { effectColour, StructureViewerPanel } from "./StructureViewer"

const originalFetch = globalThis.fetch

afterEach(() => {
  cleanup()
  globalThis.fetch = originalFetch
})

function mount() {
  return render(
    <TooltipProvider>
      <StructureViewerPanel source={{ kind: "draft", id: "draft_1" }} />
    </TooltipProvider>,
  )
}

describe("effectColour", () => {
  test("puts neutral on the middle of the ramp", () => {
    expect(effectColour(0, 2)).toBe("#f7f7f7")
  })

  test("puts the extremes on the ends", () => {
    expect(effectColour(-2, 2)).toBe("#2166ac")
    expect(effectColour(2, 2)).toBe("#b2182b")
  })

  test("clamps anything past the bound", () => {
    expect(effectColour(-40, 2)).toBe("#2166ac")
    expect(effectColour(40, 2)).toBe("#b2182b")
  })
})

describe("structure panel", () => {
  test("holds the fetch button until a full PDB ID is typed", async () => {
    mount()
    const fetchButton = screen.getByRole("button", { name: "Fetch" })
    expect(fetchButton.hasAttribute("disabled")).toBe(true)

    await userEvent.type(screen.getByLabelText("PDB ID"), "1PGA")
    expect(fetchButton.hasAttribute("disabled")).toBe(false)
  })

  test("names the third party before the wild type is sent for folding", async () => {
    mount()
    await userEvent.click(screen.getByRole("tab", { name: "ESMFold" }))
    expect(screen.getByText(/sent to ESM Atlas/)).toBeTruthy()
  })

  test("reports a rejected identifier without clearing the panel", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ detail: "no entry 9ZZZ at RCSB" }), {
        status: 422,
      })) as unknown as typeof fetch

    mount()
    await userEvent.type(screen.getByLabelText("PDB ID"), "9ZZZ")
    await userEvent.click(screen.getByRole("button", { name: "Fetch" }))

    expect(await screen.findByText("no entry 9ZZZ at RCSB")).toBeTruthy()
  })
})

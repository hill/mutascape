import { afterEach, describe, expect, test } from "bun:test"
import { streamPreview } from "./api"

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

/** Serves a fixed body one chunk at a time, so frame splitting is exercised. */
function stubStream(chunks: string[], status = 200) {
  globalThis.fetch = (async () =>
    new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          const encoder = new TextEncoder()
          for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
          controller.close()
        },
      }),
      { status, headers: { "content-type": "text/event-stream" } },
    )) as unknown as typeof fetch
}

function run(onStage: (stage: string) => void) {
  return streamPreview(
    { kind: "reference", id: "gb1_headline" },
    {} as never,
    10,
    { 1: 1 },
    onStage,
  )
}

describe("streamPreview", () => {
  test("reports every stage then resolves with the result", async () => {
    stubStream([
      'event: stage\ndata: "Loading measured data"\n\n',
      'event: stage\ndata: "Building the oracle"\n\n',
      'event: result\ndata: {"sample":{"variants":[]}}\n\n',
    ])
    const stages: string[] = []
    const result = await run((stage) => stages.push(stage))
    expect(stages).toEqual(["Loading measured data", "Building the oracle"])
    expect(result).toEqual({ sample: { variants: [] } } as never)
  })

  test("joins a frame split across two chunks", async () => {
    stubStream(['event: stage\ndata: "Sampling', ' variants"\n\n', "event: result\ndata: {}\n\n"])
    const stages: string[] = []
    await run((stage) => stages.push(stage))
    expect(stages).toEqual(["Sampling variants"])
  })

  test("raises the server detail from an error frame", async () => {
    stubStream(['event: error\ndata: {"status":404,"detail":"Unknown model."}\n\n'])
    expect(run(() => {})).rejects.toThrow("Unknown model.")
  })

  test("raises when the stream ends before a terminal frame", async () => {
    stubStream(['event: stage\ndata: "Building the oracle"\n\n'])
    expect(run(() => {})).rejects.toThrow("ended early")
  })
})

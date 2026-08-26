import type {
  CsvInspection,
  DraftResponse,
  DraftUpload,
  EvaluationResult,
  Leaderboard,
  LeaderboardEntry,
  MapScope,
  MaveDbImport,
  ModelParameters,
  ModelSummary,
  PosteriorPreviewResponse,
  PosteriorQueryResponse,
  PreviewResponse,
  PreviewSource,
  PriorMetadata,
  PublishedModel,
  ScoreHistogram,
  StructureRequestOptions,
  StructureView,
  SubstitutionMap,
} from "./types"

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...init?.headers,
    },
  })
  const body = (await response.json()) as T & { detail?: string }
  if (!response.ok) {
    throw new ApiError(
      body.detail ?? `Request failed with status ${response.status}`,
      response.status,
    )
  }
  return body
}

export async function listModels(): Promise<ModelSummary[]> {
  const body = await request<{ models: ModelSummary[] }>("/api/v1/models")
  return body.models
}

export function getModel(modelId: string): Promise<ModelSummary> {
  return request<ModelSummary>(`/api/v1/models/${modelId}`)
}

export function createDraft(upload: DraftUpload): Promise<DraftResponse> {
  return request<DraftResponse>("/api/v1/drafts", {
    method: "POST",
    body: JSON.stringify(upload),
  })
}

export function inspectCsv(dmsCsv: string): Promise<CsvInspection> {
  return request<CsvInspection>("/api/v1/csv-inspections", {
    method: "POST",
    body: JSON.stringify({ dmsCsv }),
  })
}

export function importMaveDb(reference: string): Promise<MaveDbImport> {
  return request<MaveDbImport>("/api/v1/mavedb-imports", {
    method: "POST",
    body: JSON.stringify({ reference }),
  })
}

export function fetchSubstitutionMap(
  source: PreviewSource,
  scope: MapScope = "assayed",
): Promise<SubstitutionMap> {
  return request<SubstitutionMap>("/api/v1/substitution-maps", {
    method: "POST",
    body: JSON.stringify({ source, scope }),
  })
}

export function fetchStructure(
  source: PreviewSource,
  options: StructureRequestOptions,
): Promise<StructureView> {
  return request<StructureView>("/api/v1/structures", {
    method: "POST",
    body: JSON.stringify({ source, ...options }),
  })
}

export function fetchScoreHistogram(source: PreviewSource): Promise<ScoreHistogram> {
  return request<ScoreHistogram>("/api/v1/score-histograms", {
    method: "POST",
    body: JSON.stringify({ source }),
  })
}

export function runPreview(
  source: PreviewSource,
  parameters: ModelParameters,
  sampleSize: number,
  depthWeights: Record<number, number>,
): Promise<PreviewResponse> {
  return request<PreviewResponse>("/api/v1/previews", {
    method: "POST",
    body: JSON.stringify({
      source,
      parameters,
      sample: { size: sampleSize, depthWeights },
    }),
  })
}

/**
 * Reads a Server-Sent Event stream of build stages, resolving on the terminal frame.
 * The server sends one `stage` event per phase, then a single `result` or `error`.
 */
async function streamStages<T>(
  path: string,
  body: unknown,
  onStage: (stage: string) => void,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!response.ok || !response.body) {
    throw new ApiError(`Request failed with status ${response.status}`, response.status)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let outcome: { value: T } | null = null

  while (outcome === null) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // Frames are separated by a blank line, so anything after the last one is a partial
    // frame and has to stay in the buffer.
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const event = frame.match(/^event: (.*)$/m)?.[1]
      const data = frame.match(/^data: (.*)$/m)?.[1]
      if (!event || data === undefined) continue
      const parsed = JSON.parse(data)
      if (event === "stage") onStage(parsed as string)
      if (event === "result") outcome = { value: parsed as T }
      if (event === "error") {
        throw new ApiError(parsed.detail ?? "The build failed.", parsed.status ?? 500)
      }
    }
  }

  if (outcome === null) throw new ApiError("The build stream ended early.", 500)
  return outcome.value
}

export function streamPreview(
  source: PreviewSource,
  parameters: ModelParameters,
  sampleSize: number,
  depthWeights: Record<number, number>,
  onStage: (stage: string) => void,
): Promise<PreviewResponse> {
  return streamStages<PreviewResponse>(
    "/api/v1/preview-events",
    { source, parameters, sample: { size: sampleSize, depthWeights } },
    onStage,
  )
}

export function streamPosteriorPreview(
  source: PreviewSource,
  priorId: string,
  seed: number,
  ensembleSize: number,
  sampleSize: number,
  depthWeights: Record<number, number>,
  onStage: (stage: string) => void,
): Promise<PosteriorPreviewResponse> {
  return streamStages<PosteriorPreviewResponse>(
    "/api/v1/posterior-preview-events",
    { source, priorId, seed, ensembleSize, sample: { size: sampleSize, depthWeights } },
    onStage,
  )
}

export async function evaluateLandscape(
  source: PreviewSource,
  parameters: ModelParameters,
  mutationLabel: string,
): Promise<EvaluationResult> {
  const body = await request<{ results: EvaluationResult[] }>("/api/v1/evaluations", {
    method: "POST",
    body: JSON.stringify({
      source,
      parameters,
      entries: [{ mutationLabel }],
    }),
  })
  return body.results[0]
}

export function publishModel(
  draftId: string,
  name: string,
  parameters: ModelParameters,
  options?: {
    modelType: "deterministic" | "posterior"
    priorId: string
    oracleSeed: number
  },
): Promise<PublishedModel> {
  return request<PublishedModel>("/api/v1/models", {
    method: "POST",
    body: JSON.stringify({ draftId, name, parameters, ...options }),
  })
}

export async function listPriors(): Promise<{
  priors: PriorMetadata[]
  defaultPriorId: string
}> {
  return request("/api/v1/priors")
}

export function queryPosterior(
  source: PreviewSource,
  priorId: string,
  seed: number,
  ensembleSize: number,
  mutationLabel: string,
): Promise<PosteriorQueryResponse> {
  return request<PosteriorQueryResponse>("/api/v1/posterior-queries", {
    method: "POST",
    body: JSON.stringify({
      source,
      priorId,
      seed,
      ensembleSize,
      entries: [{ mutationLabel }],
    }),
  })
}

export function runPosteriorPreview(
  source: PreviewSource,
  priorId: string,
  seed: number,
  ensembleSize: number,
  sampleSize: number,
  depthWeights: Record<number, number>,
): Promise<PosteriorPreviewResponse> {
  return request<PosteriorPreviewResponse>("/api/v1/posterior-previews", {
    method: "POST",
    body: JSON.stringify({
      source,
      priorId,
      seed,
      ensembleSize,
      sample: { size: sampleSize, depthWeights },
    }),
  })
}

export async function listLeaderboards(): Promise<Leaderboard[]> {
  const body = await request<{ leaderboards: Leaderboard[] }>("/api/v1/leaderboards")
  return body.leaderboards
}

export function createLeaderboard(
  name: string,
  landscapeId: string,
  queryBudget: number,
): Promise<Leaderboard> {
  return request<Leaderboard>("/api/v1/leaderboards", {
    method: "POST",
    body: JSON.stringify({ name, landscapeId, queryBudget }),
  })
}

export function addLeaderboardEntry(
  boardId: string,
  modelName: string,
  successes: number,
  trials: number,
): Promise<LeaderboardEntry> {
  return request<LeaderboardEntry>(`/api/v1/leaderboards/${boardId}/entries`, {
    method: "POST",
    body: JSON.stringify({ modelName, successes, trials }),
  })
}

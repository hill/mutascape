import type {
  DraftResponse,
  DraftUpload,
  EvaluationResult,
  Leaderboard,
  LeaderboardEntry,
  ModelParameters,
  ModelSummary,
  PosteriorPreviewResponse,
  PosteriorQueryResponse,
  PreviewResponse,
  PreviewSource,
  PriorMetadata,
  PublishedModel,
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

export interface ModelParameters {
  noiseSigma: number
  alpha: number
  pairwiseSigma: number
  combinatorialUpside: number
  seed: number
}

export type DmsSchema = "singles" | "variants" | "hgvs"

export interface SourceSummary {
  wtLength: number
  measuredRows: number
  hasStructure: boolean
  contactCount: number
  canFitNonlinearity: boolean
  structureMethod?: "none" | "uploaded" | "esmfold"
  observationCount?: number
  higherOrderRows?: number
  dmsSchema?: DmsSchema
  wildTypeFitness?: number
  supportedSubstitutions?: number
  previewCapacity?: number
}

export interface PosteriorSettings {
  priorId: string
  oracleSeed: number
}

export interface ModelSummary {
  id: string
  name: string
  kind: "reference" | "custom" | "posterior"
  description: string
  positions: number[]
  wtCompact: string
  parameters: ModelParameters
  createdAt: string | null
  sourceSummary?: SourceSummary
  examples?: CodeExamples
  posteriorSettings?: PosteriorSettings
}

export interface DraftSummary {
  skippedRows: Record<string, number>
  wildTypeScoreAssumed: boolean
  positions: number[]
  positionCount: number
  wtLength: number
  wtCompact: string
  measuredRows: number
  imputedEffects: number
  canFitNonlinearity: boolean
  hasStructure: boolean
  resolvedResidues: number
  contactCount: number
  structureMethod: "none" | "uploaded" | "esmfold"
  observationCount: number
  higherOrderRows: number
  dmsSchema: DmsSchema
  wildTypeFitness: number
  supportedSubstitutions: number
  previewCapacity: number
}

export interface DraftResponse {
  id: string
  expiresAt: string
  summary: DraftSummary
}

export type MapScope = "assayed" | "sequence"

export interface SubstitutionMap {
  positions: number[]
  aminoAcids: string
  wildTypeCompact: string
  assayed: number[]
  measured: number[][]
  imputed: number[][]
  wildTypeCells: number[][]
  measuredCount: number
  imputedCount: number
  lower: number
  upper: number
  bound: number
  scope: MapScope
  assayedCount: number
  sequenceLength: number
  totalPositions: number
  truncated: boolean
}

export type StructureSource = "rcsb" | "upload" | "esmfold"

export interface StructureResidue {
  position: number
  residue: number
  parent: string
  structureResidue: string
  matched: boolean
  effect: number | null
}

export interface StructureView {
  pdb: string
  format: "pdb" | "mmcif"
  chains: string[]
  chainId: string
  chainLength: number
  residueCount: number
  identity: number
  bound: number
  residues: StructureResidue[]
  unmatched: number[]
  mismatched: number[]
  method: StructureSource
  label: string
}

export interface StructureRequestOptions {
  kind: StructureSource
  pdbId?: string
  structureText?: string
  structureFormat?: string
  chainId?: string
}

export interface ScoreHistogram {
  centres: number[]
  binWidth: number
  singles: number[]
  higherOrder: number[]
  singleCount: number
  higherOrderCount: number
  wildType: number
  median: number
  min: number
  max: number
}

export interface CsvSchemaRole {
  id: string
  label: string
  guess: string | null
}

export interface CsvSchemaOption {
  id: DmsSchema
  label: string
  complete: boolean
  roles: CsvSchemaRole[]
}

export interface ColumnMapping {
  schema: DmsSchema
  columns: Record<string, string>
}

export interface CsvInspection {
  headers: string[]
  rowCount: number
  sampleRows: string[][]
  schemas: CsvSchemaOption[]
  detected: ColumnMapping | null
}

export interface MaveDbImport {
  urn: string
  title: string
  shortDescription: string
  targetName: string
  recordUrl: string
  numVariants: number
  license: string
  publications: string[]
  wtFasta: string
  dmsCsv: string
  inspection: CsvInspection
}

export interface ProfilePoint {
  depth: number
  count: number
  q05: number
  q25: number
  median: number
  q75: number
  q95: number
}

export interface SampleRow {
  sequence: string
  variant: string
  fitness: number
  depth: number
  measuredFitness: number | null
}

export interface SamplePreview {
  requested: number
  returnedRows: number
  depthWeights: Record<string, number>
  profile: ProfilePoint[]
  rows: SampleRow[]
  graph: LandscapeGraph
}

export interface LandscapeGraphNode {
  id: string
  variant: string
  depth: number
  fitness: number
  sampled: boolean
  posteriorMean?: number
  posteriorStd?: number
}

export interface LandscapeGraphEdge {
  source: string
  target: string
}

export interface LandscapeGraph {
  nodes: LandscapeGraphNode[]
  edges: LandscapeGraphEdge[]
}

export interface MetricRow {
  name: string
  real: number | null
  synthetic: number
}

export interface WalshPreview {
  orders: number[]
  real: Record<string, number>
  synthetic: Record<string, number>
}

export interface PreviewResponse {
  source: Pick<ModelSummary, "id" | "name" | "kind" | "positions" | "wtCompact">
  parameters: ModelParameters
  comparisonAvailable: boolean
  profiles: {
    real?: ProfilePoint[]
    synthetic: ProfilePoint[]
  }
  walsh: WalshPreview | null
  metrics: MetricRow[]
  sample: SamplePreview
  notices: string[]
  cached: boolean
}

export interface CodeExamples {
  localPython: string
  hostedPython: string
}

export interface EvaluationResult {
  index: number
  mutationLabel: string
  fitness: number
}

export interface PublishedModel extends ModelSummary {
  created: boolean
  examples: CodeExamples
}

export interface DraftUpload {
  dmsCsv: string
  wtFasta: string
  structureText?: string
  structureFormat?: "pdb" | "cif" | "mmcif"
  structureInference?: "none" | "esmfold"
  columnMap?: ColumnMapping
  wildTypeScore?: number
}

export type PreviewSource = {
  kind: "reference" | "draft" | "model"
  id: string
}

export interface LeaderboardEntry {
  id: string
  modelName: string
  successes: number
  trials: number
  successRate: number
  confidenceLow: number
  confidenceHigh: number
  createdAt: string
}

export interface Leaderboard {
  id: string
  name: string
  landscapeId: string
  queryBudget: number
  createdAt: string
  entries: LeaderboardEntry[]
}

export interface PriorSource {
  name: string
  url: string
  revision: string
  landscapes: number
}

export interface PriorMetadata {
  id: string
  name: string
  description: string
  created: string
  maxInteractionOrder: number
  nLandscapes: number
  nProteins: number
  sources: PriorSource[]
}

export interface PosteriorInterval {
  lower: number
  upper: number
}

export interface PosteriorQueryResult {
  index: number
  mutationLabel: string
  depth: number
  observed: boolean
  additiveFitness: number
  fixedOracleFitness: number
  posteriorMean: number
  posteriorStd: number
  intervals: Record<"50" | "80" | "95", PosteriorInterval>
  samples: number[]
}

export interface PosteriorQueryResponse {
  source: Pick<ModelSummary, "id" | "name" | "kind" | "positions" | "wtCompact">
  prior: PriorMetadata
  settings: { seed: number; ensembleSize: number }
  results: PosteriorQueryResult[]
}

export interface PosteriorProfilePoint extends ProfilePoint {
  fixedMedian: number
  additiveMedian: number
  meanStd: number
}

export interface PosteriorSampleRow {
  sequence: string
  variant: string
  depth: number
  additiveFitness: number
  fixedOracleFitness: number
  posteriorMean: number
  posteriorStd: number
  lower80: number
  upper80: number
}

export interface PosteriorPreviewResponse {
  source: Pick<ModelSummary, "id" | "name" | "kind" | "positions" | "wtCompact">
  prior: PriorMetadata
  settings: {
    seed: number
    ensembleSize: number
    generatorKind: string
    coupling: number
  }
  profile: PosteriorProfilePoint[]
  sample: {
    requested: number
    returnedRows: number
    depthWeights: Record<string, number>
    rows: PosteriorSampleRow[]
    graph: LandscapeGraph
  }
  notices: string[]
}

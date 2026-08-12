export interface ModelParameters {
  noiseSigma: number;
  alpha: number;
  pairwiseSigma: number;
  combinatorialUpside: number;
  seed: number;
}

export interface SourceSummary {
  wtLength: number;
  measuredRows: number;
  hasStructure: boolean;
  contactCount: number;
  canFitNonlinearity: boolean;
  structureMethod?: "none" | "uploaded" | "esmfold";
}

export interface ModelSummary {
  id: string;
  name: string;
  kind: "reference" | "custom";
  description: string;
  positions: number[];
  wtCompact: string;
  parameters: ModelParameters;
  createdAt: string | null;
  sourceSummary?: SourceSummary;
  examples?: CodeExamples;
}

export interface DraftSummary {
  positions: number[];
  positionCount: number;
  wtLength: number;
  wtCompact: string;
  measuredRows: number;
  imputedEffects: number;
  canFitNonlinearity: boolean;
  hasStructure: boolean;
  resolvedResidues: number;
  contactCount: number;
  structureMethod: "none" | "uploaded" | "esmfold";
}

export interface DraftResponse {
  id: string;
  expiresAt: string;
  summary: DraftSummary;
}

export interface ProfilePoint {
  depth: number;
  count: number;
  q05: number;
  q25: number;
  median: number;
  q75: number;
  q95: number;
}

export interface SampleRow {
  sequence: string;
  variant: string;
  fitness: number;
  depth: number;
  measuredFitness: number | null;
}

export interface SamplePreview {
  requested: number;
  returnedRows: number;
  depthWeights: Record<string, number>;
  profile: ProfilePoint[];
  rows: SampleRow[];
  graph: LandscapeGraph;
}

export interface LandscapeGraphNode {
  id: string;
  variant: string;
  depth: number;
  fitness: number;
  sampled: boolean;
}

export interface LandscapeGraphEdge {
  source: string;
  target: string;
}

export interface LandscapeGraph {
  nodes: LandscapeGraphNode[];
  edges: LandscapeGraphEdge[];
}

export interface MetricRow {
  name: string;
  real: number | null;
  synthetic: number;
}

export interface WalshPreview {
  orders: number[];
  real: Record<string, number>;
  synthetic: Record<string, number>;
}

export interface PreviewResponse {
  source: Pick<ModelSummary, "id" | "name" | "kind" | "positions" | "wtCompact">;
  parameters: ModelParameters;
  comparisonAvailable: boolean;
  profiles: {
    real?: ProfilePoint[];
    synthetic: ProfilePoint[];
  };
  walsh: WalshPreview | null;
  metrics: MetricRow[];
  sample: SamplePreview;
  notices: string[];
  cached: boolean;
}

export interface CodeExamples {
  localPython: string;
  hostedPython: string;
}

export interface EvaluationResult {
  index: number;
  mutationLabel: string;
  fitness: number;
}

export interface PublishedModel extends ModelSummary {
  created: boolean;
  examples: CodeExamples;
}

export interface DraftUpload {
  dmsCsv: string;
  wtFasta: string;
  structureText?: string;
  structureFormat?: "pdb" | "cif" | "mmcif";
  structureInference?: "none" | "esmfold";
}

export type PreviewSource = {
  kind: "reference" | "draft" | "model";
  id: string;
};

export interface LeaderboardEntry {
  id: string;
  modelName: string;
  successes: number;
  trials: number;
  successRate: number;
  confidenceLow: number;
  confidenceHigh: number;
  createdAt: string;
}

export interface Leaderboard {
  id: string;
  name: string;
  landscapeId: string;
  queryBudget: number;
  createdAt: string;
  entries: LeaderboardEntry[];
}

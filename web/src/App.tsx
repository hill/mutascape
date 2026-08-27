import {
  Badge,
  Banner,
  Button,
  Chart,
  ClipboardText,
  Dialog,
  Empty,
  Grid,
  Input,
  type KumoChartOption,
  LayerCard,
  Select,
  Sidebar,
  Table,
  Tabs,
  Text,
  Tooltip,
} from "@cloudflare/kumo"
import {
  ArrowClockwise,
  ArrowLeft,
  BookOpenText,
  ChartBar,
  CheckCircle,
  Database,
  Dna,
  DownloadSimple as Download,
  Flask,
  Info,
  Minus,
  Play,
  Plus,
  SquaresFour,
  Trophy,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react"
import { BarChart, GraphChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts"
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components"
import * as echarts from "echarts/core"
import { CanvasRenderer } from "echarts/renderers"
import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react"
import {
  addLeaderboardEntry,
  createDraft,
  createLeaderboard,
  evaluateLandscape,
  fetchScoreHistogram,
  fetchSubstitutionMap,
  getModel,
  importMaveDb,
  inspectCsv,
  listLeaderboards,
  listModels,
  publishModel,
  queryPosterior,
  streamPosteriorPreview,
  streamPreview,
} from "./api"
import {
  fitnessProfileOptions,
  landscapeGraphOptions,
  posteriorProfileOptions,
  reachableFrom,
  scoreHistogramOptions,
  substitutionMapOptions,
  walshOptions,
} from "./charts"
import DocumentationSite, { type DocsPage, isDocsPage } from "./docs/DocumentationSite"
import { highlightPython } from "./python-highlight"
import { StructureViewerPanel } from "./StructureViewer"
import type {
  CodeExamples,
  ColumnMapping,
  CsvInspection,
  DraftResponse,
  DraftSummary,
  EvaluationResult,
  LandscapeGraph,
  Leaderboard,
  LeaderboardEntry,
  MapScope,
  MaveDbImport,
  ModelParameters,
  ModelSummary,
  PosteriorPreviewResponse,
  PosteriorQueryResult,
  PreviewResponse,
  PreviewSource,
  ScoreHistogram,
  SubstitutionMap,
} from "./types"

echarts.use([
  LineChart,
  BarChart,
  GraphChart,
  HeatmapChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  DataZoomComponent,
  AriaComponent,
  CanvasRenderer,
])

const DEFAULT_PARAMETERS: ModelParameters = {
  noiseSigma: 0.3,
  alpha: 1.5,
  pairwiseSigma: 0,
  combinatorialUpside: 3,
  seed: 42,
}

export type Page = "studio" | "explore" | "models" | "leaderboards" | DocsPage

interface AppProps {
  page?: Page
  landscapeId?: string
  onPageChange?: (page: Page) => void
  onExploreLandscape?: (modelId: string) => void
}

function sourceFromModel(model: ModelSummary, draftId?: string): PreviewSource {
  if (draftId && model.id === draftId) return { kind: "draft", id: model.id }
  const isBundled = model.kind === "reference" || (model.kind === "posterior" && !model.createdAt)
  return { kind: isBundled ? "reference" : "model", id: model.id }
}

function initialWeights(maxDepth: number): Record<number, number> {
  return Object.fromEntries(Array.from({ length: maxDepth }, (_, index) => [index + 1, 1]))
}

// Only depths carrying a positive weight are drawn from, so a zero share removes that
// depth's genotypes from the pool. Counting them would let the form request more unique
// variants than exist and the sampler rejects the whole plan.
function sampleCapacity(positionCount: number, depths: number[]): number {
  let total = 0
  for (const depth of depths) {
    if (depth < 1 || depth > positionCount) continue
    let combinations = 1
    for (let step = 1; step <= depth; step++) {
      combinations = (combinations * (positionCount - step + 1)) / step
    }
    total += combinations * 19 ** depth
  }
  return Math.min(100_000, Math.floor(total))
}

function activeDepths(depthWeights: Record<number, number>): number[] {
  const depths = Object.entries(depthWeights)
    .filter(([, weight]) => weight > 0)
    .map(([depth]) => Number(depth))
  return depths.length > 0 ? depths : [1]
}

function modelSampleLimit(
  model: ModelSummary | null,
  depthWeights: Record<number, number>,
): number {
  if (!model) return 1
  const theoretical = sampleCapacity(model.positions.length, activeDepths(depthWeights))
  if (model.kind !== "posterior") return Math.max(1, theoretical)
  const measured = model.sourceSummary?.previewCapacity ?? theoretical
  return Math.max(1, Math.min(5_000, measured, theoretical))
}

function formatNumber(value: number | null, metricName?: string): string {
  if (value === null) return "N/A"
  if (metricName === "Local optima") return Math.round(value).toLocaleString()
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2)
  return value.toFixed(3)
}

export default function App({
  page = "studio",
  landscapeId,
  onPageChange,
  onExploreLandscape,
}: AppProps) {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [draft, setDraft] = useState<DraftResponse | null>(null)
  const [draftModel, setDraftModel] = useState<ModelSummary | null>(null)
  const [selectedId, setSelectedId] = useState("")
  const [parameters, setParameters] = useState<ModelParameters>(DEFAULT_PARAMETERS)
  const [sampleSize, setSampleSize] = useState(10_000)
  const [maxDepth, setMaxDepth] = useState(4)
  const [depthWeights, setDepthWeights] = useState<Record<number, number>>(initialWeights(4))
  const [preview, setPreview] = useState<PreviewResponse | null>(null)
  const [posteriorPreview, setPosteriorPreview] = useState<PosteriorPreviewResponse | null>(null)
  const [priorId, setPriorId] = useState("combinatorial-v2")
  const [oracleSeed, setOracleSeed] = useState(42)
  const [ensembleSize, setEnsembleSize] = useState(100)
  const [loadingModels, setLoadingModels] = useState(true)
  const [running, setRunning] = useState(false)
  const [stages, setStages] = useState<BuildStage[]>([])
  const [error, setError] = useState<string | null>(null)
  const [publishedExamples, setPublishedExamples] = useState<CodeExamples | null>(null)
  const [substitutionMap, setSubstitutionMap] = useState<SubstitutionMap | null>(null)
  const [mapScope, setMapScope] = useState<MapScope>("assayed")
  const [scoreHistogram, setScoreHistogram] = useState<ScoreHistogram | null>(null)

  const availableModels = useMemo(
    () => (draftModel ? [draftModel, ...models] : models),
    [draftModel, models],
  )
  const selectedModel = availableModels.find((model) => model.id === selectedId) ?? null
  const selectedIsDraft = draft !== null && selectedId === draft.id
  const isDocumentation = isDocsPage(page)
  const selectedIsPosterior = selectedModel?.kind === "posterior"
  const parametersLocked = selectedModel?.kind === "custom" && !selectedIsDraft
  const hasStructure =
    selectedModel?.kind === "reference" || selectedModel?.sourceSummary?.hasStructure === true
  const canFitNonlinearity =
    selectedModel?.kind === "reference" || selectedModel?.sourceSummary?.canFitNonlinearity === true

  const refreshModels = useCallback(async () => {
    setLoadingModels(true)
    try {
      const items = await listModels()
      setModels(items)
      // An empty selection is the starting state, not a stale one, so it survives a
      // refresh. Only an id that has disappeared from the catalogue gets repointed.
      setSelectedId((current) =>
        current && current !== draft?.id && !items.some((model) => model.id === current)
          ? (items[0]?.id ?? "")
          : current,
      )
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load the model catalogue.")
    } finally {
      setLoadingModels(false)
    }
  }, [draft?.id])

  useEffect(() => {
    void refreshModels()
  }, [refreshModels])

  useEffect(() => {
    if (isDocumentation) {
      setPublishedExamples(null)
      return
    }
    if (!selectedId || selectedIsDraft) {
      setPublishedExamples(null)
      return
    }
    let active = true
    void getModel(selectedId)
      .then((model) => {
        if (active) setPublishedExamples(model.examples ?? null)
      })
      .catch(() => {
        if (active) setPublishedExamples(null)
      })
    return () => {
      active = false
    }
  }, [isDocumentation, selectedId, selectedIsDraft])

  useEffect(() => {
    setSubstitutionMap(null)
    if (isDocumentation || !selectedModel) return
    let active = true
    void fetchSubstitutionMap(sourceFromModel(selectedModel, draft?.id), mapScope)
      .then((map) => {
        if (active) setSubstitutionMap(map)
      })
      .catch(() => {
        // A missing grid leaves the panel hidden. It must never block the build flow.
        if (active) setSubstitutionMap(null)
      })
    return () => {
      active = false
    }
  }, [isDocumentation, selectedModel, draft?.id, mapScope])

  useEffect(() => {
    setScoreHistogram(null)
    if (isDocumentation || !selectedModel) return
    let active = true
    void fetchScoreHistogram(sourceFromModel(selectedModel, draft?.id))
      .then((histogram) => {
        if (active) setScoreHistogram(histogram)
      })
      .catch(() => {
        if (active) setScoreHistogram(null)
      })
    return () => {
      active = false
    }
  }, [isDocumentation, selectedModel, draft?.id])

  useEffect(() => {
    if (!landscapeId || landscapeId === selectedId) return
    const model = availableModels.find((item) => item.id === landscapeId)
    if (!model) return
    setSelectedId(model.id)
    setParameters(model.parameters)
    setPriorId(model.posteriorSettings?.priorId ?? "combinatorial-v2")
    setOracleSeed(model.posteriorSettings?.oracleSeed ?? 42)
    const nextMaxDepth = Math.max(1, Math.min(4, model.positions.length))
    setMaxDepth(nextMaxDepth)
    setDepthWeights(initialWeights(nextMaxDepth))
    setSampleSize(
      Math.min(
        model.kind === "posterior" ? 500 : 10_000,
        modelSampleLimit(model, initialWeights(nextMaxDepth)),
      ),
    )
    setPreview(null)
    setPosteriorPreview(null)
    setError(null)
  }, [availableModels, landscapeId, selectedId])

  function selectModel(modelId: string) {
    const model = availableModels.find((item) => item.id === modelId)
    if (!model) return
    setSelectedId(modelId)
    setParameters(model.parameters)
    setPriorId(model.posteriorSettings?.priorId ?? "combinatorial-v2")
    setOracleSeed(model.posteriorSettings?.oracleSeed ?? 42)
    const nextMaxDepth = Math.max(1, Math.min(4, model.positions.length))
    setMaxDepth(nextMaxDepth)
    setDepthWeights(initialWeights(nextMaxDepth))
    setSampleSize(
      Math.min(
        model.kind === "posterior" ? 500 : 10_000,
        modelSampleLimit(model, initialWeights(nextMaxDepth)),
      ),
    )
    setPreview(null)
    setPosteriorPreview(null)
    setError(null)
  }

  function changeMaxDepth(value: number) {
    const positionCount = selectedModel?.positions.length ?? 1
    const bounded = Math.max(1, Math.min(positionCount, Math.floor(value || 1)))
    setMaxDepth(bounded)
    const next: Record<number, number> = {}
    for (let depth = 1; depth <= bounded; depth++) next[depth] = depthWeights[depth] ?? 1
    setDepthWeights(next)
    setSampleSize((current) => Math.min(current, modelSampleLimit(selectedModel, next)))
  }

  function changeDepthWeights(next: Record<number, number>) {
    setDepthWeights(next)
    setSampleSize((current) => Math.min(current, modelSampleLimit(selectedModel, next)))
  }

  async function applyPreview(): Promise<boolean> {
    if (!selectedModel) return false
    setRunning(true)
    setError(null)
    setStages([])
    const startStage = (stage: string) => {
      const at = performance.now()
      setStages((current) => [
        ...current.map((entry) => (entry.endedAt === null ? { ...entry, endedAt: at } : entry)),
        { name: stage, startedAt: at, endedAt: null },
      ])
    }
    const finishStages = () => {
      const at = performance.now()
      setStages((current) =>
        current.map((entry) => (entry.endedAt === null ? { ...entry, endedAt: at } : entry)),
      )
    }
    try {
      const source = sourceFromModel(selectedModel, draft?.id)
      if (selectedIsPosterior) {
        const result = await streamPosteriorPreview(
          source,
          priorId,
          oracleSeed,
          ensembleSize,
          sampleSize,
          depthWeights,
          startStage,
        )
        setPosteriorPreview(result)
        setPreview(null)
      } else {
        const result = await streamPreview(source, parameters, sampleSize, depthWeights, startStage)
        setPreview(result)
        setPosteriorPreview(null)
      }
      finishStages()
      return true
    } catch (cause) {
      finishStages()
      setError(cause instanceof Error ? cause.message : "The preview failed.")
      return false
    } finally {
      setRunning(false)
    }
  }

  async function createLandscape() {
    if (!selectedModel) return
    if (await applyPreview()) onExploreLandscape?.(selectedModel.id)
  }

  function acceptDraft(created: DraftResponse, name: string) {
    const summary = created.summary
    const model: ModelSummary = {
      id: created.id,
      name: name || "Custom draft",
      kind: "posterior",
      description: "Temporary DMS-pairwise benchmark oracle.",
      positions: summary.positions,
      wtCompact: summary.wtCompact,
      parameters: DEFAULT_PARAMETERS,
      createdAt: null,
      sourceSummary: {
        wtLength: summary.wtLength,
        measuredRows: summary.measuredRows,
        hasStructure: summary.hasStructure,
        contactCount: summary.contactCount,
        canFitNonlinearity: summary.canFitNonlinearity,
        structureMethod: summary.structureMethod,
        observationCount: summary.observationCount,
        higherOrderRows: summary.higherOrderRows,
        dmsSchema: summary.dmsSchema,
        wildTypeFitness: summary.wildTypeFitness,
        supportedSubstitutions: summary.supportedSubstitutions,
        previewCapacity: summary.previewCapacity,
      },
      posteriorSettings: { priorId: "combinatorial-v2", oracleSeed: 42 },
    }
    setDraft(created)
    setDraftModel(model)
    setSelectedId(created.id)
    setParameters(DEFAULT_PARAMETERS)
    setPriorId("combinatorial-v2")
    setOracleSeed(42)
    const nextMaxDepth = Math.max(1, Math.min(4, summary.positionCount))
    setMaxDepth(nextMaxDepth)
    setDepthWeights(initialWeights(nextMaxDepth))
    setSampleSize(Math.min(500, modelSampleLimit(model, initialWeights(nextMaxDepth))))
    setPreview(null)
    setPosteriorPreview(null)
    setError(null)
    onPageChange?.("studio")
  }

  async function publish(name: string) {
    if (!draft) return
    const model = await publishModel(draft.id, name, parameters, {
      modelType: "posterior",
      priorId,
      oracleSeed,
    })
    setPublishedExamples(model.examples)
    await refreshModels()
    setSelectedId(model.id)
    setDraft(null)
    setDraftModel(null)
    onExploreLandscape?.(model.id)
  }

  return (
    <Sidebar.Provider
      contained
      defaultOpen
      variant="inset"
      collapsible="icon"
      resizable
      defaultWidth={250}
      minWidth={210}
      maxWidth={340}
      className="app-shell"
    >
      <AppSidebar
        page={page}
        models={availableModels}
        selectedId={selectedId}
        loading={loadingModels}
        onPageChange={onPageChange}
        onSelectLandscape={(modelId) => {
          selectModel(modelId)
          onExploreLandscape?.(modelId)
        }}
      />

      <section className="main-window">
        <header className="window-bar">
          <div className="window-context">
            <Sidebar.Trigger />
            <Text size="sm" bold>
              {page === "studio" && "Create landscape"}
              {page === "explore" && (selectedModel?.name ?? "Explore landscape")}
              {page === "models" && "Landscape library"}
              {page === "leaderboards" && "Leaderboards"}
              {isDocumentation && "Documentation"}
            </Text>
          </div>
          <Badge className="research-preview-badge" variant="beta">
            Research preview
          </Badge>
        </header>

        <main className={isDocumentation ? "page-frame docs-page-host" : "page-frame"}>
          {error && (
            <Banner
              variant="error"
              icon={<WarningCircle weight="fill" />}
              title="Request failed"
              description={error}
            />
          )}

          {page === "studio" && (
            <LandscapeBuilder
              models={availableModels}
              selectedModel={selectedModel}
              loadingModels={loadingModels}
              selectModel={selectModel}
              parameters={parameters}
              setParameters={setParameters}
              parametersLocked={parametersLocked}
              hasStructure={hasStructure}
              canFitNonlinearity={canFitNonlinearity}
              priorId={priorId}
              setPriorId={setPriorId}
              oracleSeed={oracleSeed}
              setOracleSeed={setOracleSeed}
              ensembleSize={ensembleSize}
              setEnsembleSize={setEnsembleSize}
              running={running}
              stages={stages}
              createLandscape={createLandscape}
              acceptDraft={acceptDraft}
              intakeReport={draft && draft.id === selectedId ? draft.summary : null}
              substitutionMap={substitutionMap}
              mapScope={mapScope}
              setMapScope={setMapScope}
              scoreHistogram={scoreHistogram}
              structureSource={selectedModel ? sourceFromModel(selectedModel, draft?.id) : null}
            />
          )}
          {page === "explore" && (
            <LandscapeExplorer
              selectedModel={selectedModel}
              parameters={parameters}
              priorId={priorId}
              oracleSeed={oracleSeed}
              ensembleSize={ensembleSize}
              sampleSize={sampleSize}
              sampleLimit={modelSampleLimit(selectedModel, depthWeights)}
              setSampleSize={setSampleSize}
              maxDepth={maxDepth}
              changeMaxDepth={changeMaxDepth}
              depthWeights={depthWeights}
              setDepthWeights={changeDepthWeights}
              running={running}
              stages={stages}
              applyPreview={applyPreview}
              draft={draft}
              draftName={draftModel?.name ?? ""}
              publish={publish}
              preview={preview}
              posteriorPreview={posteriorPreview}
              publishedExamples={publishedExamples}
              onBack={() => onPageChange?.("studio")}
            />
          )}
          {page === "models" && (
            <ModelCatalogue
              models={models}
              loading={loadingModels}
              onRefresh={refreshModels}
              onExplore={(modelId) => {
                selectModel(modelId)
                onExploreLandscape?.(modelId)
              }}
            />
          )}
          {page === "leaderboards" && <Leaderboards models={models} />}
          {isDocsPage(page) && (
            <DocumentationSite
              page={page}
              onNavigate={(nextPage) => onPageChange?.(nextPage)}
              onOpenStudio={() => onPageChange?.("studio")}
            />
          )}
        </main>
      </section>
    </Sidebar.Provider>
  )
}

function AppSidebar({
  page,
  models,
  selectedId,
  loading,
  onPageChange,
  onSelectLandscape,
}: {
  page: Page
  models: ModelSummary[]
  selectedId: string
  loading: boolean
  onPageChange?: (page: Page) => void
  onSelectLandscape: (modelId: string) => void
}) {
  return (
    <Sidebar className="app-sidebar" contentClassName="app-sidebar-content">
      <Sidebar.Header className="sidebar-brand">
        <div className="sidebar-logo">
          <img src="/assets/logo.png" alt="mutascape" />
        </div>
      </Sidebar.Header>
      <Sidebar.Content>
        <Sidebar.Group>
          <Sidebar.GroupLabel>Workspace</Sidebar.GroupLabel>
          <Sidebar.Menu>
            <Sidebar.MenuButton
              size="sm"
              icon={Plus}
              active={page === "studio"}
              tooltip="Create landscape"
              onClick={() => onPageChange?.("studio")}
            >
              Create landscape
            </Sidebar.MenuButton>
            <Sidebar.MenuButton
              size="sm"
              icon={SquaresFour}
              active={page === "models"}
              tooltip="Landscape library"
              onClick={() => onPageChange?.("models")}
            >
              Landscape library
            </Sidebar.MenuButton>
            <Sidebar.MenuButton
              size="sm"
              icon={Trophy}
              active={page === "leaderboards"}
              tooltip="Leaderboards"
              onClick={() => onPageChange?.("leaderboards")}
            >
              Leaderboards
            </Sidebar.MenuButton>
            <Sidebar.MenuButton
              size="sm"
              icon={BookOpenText}
              active={isDocsPage(page)}
              tooltip="Documentation"
              onClick={() => onPageChange?.("docs")}
            >
              Documentation
            </Sidebar.MenuButton>
          </Sidebar.Menu>
        </Sidebar.Group>
        <Sidebar.Separator />
        <Sidebar.Group>
          <Sidebar.GroupLabel>Landscapes</Sidebar.GroupLabel>
          {loading ? (
            <Sidebar.Loading label="Loading landscapes" />
          ) : (
            <Sidebar.Menu>
              {models.map((model) => (
                <Sidebar.MenuButton
                  key={model.id}
                  size="sm"
                  icon={Dna}
                  active={page === "explore" && model.id === selectedId}
                  tooltip={model.name}
                  onClick={() => onSelectLandscape(model.id)}
                >
                  {model.name}
                </Sidebar.MenuButton>
              ))}
            </Sidebar.Menu>
          )}
          <Sidebar.Menu>
            <Sidebar.MenuButton
              size="sm"
              icon={Plus}
              tooltip="New landscape"
              onClick={() => onPageChange?.("studio")}
            >
              New landscape
            </Sidebar.MenuButton>
          </Sidebar.Menu>
        </Sidebar.Group>
      </Sidebar.Content>
      <Sidebar.ResizeHandle />
    </Sidebar>
  )
}

interface LandscapeBuilderProps {
  models: ModelSummary[]
  structureSource: PreviewSource | null
  selectedModel: ModelSummary | null
  loadingModels: boolean
  selectModel: (id: string) => void
  parameters: ModelParameters
  setParameters: (value: ModelParameters) => void
  parametersLocked: boolean
  hasStructure: boolean
  canFitNonlinearity: boolean
  priorId: string
  setPriorId: (value: string) => void
  oracleSeed: number
  setOracleSeed: (value: number) => void
  ensembleSize: number
  setEnsembleSize: (value: number) => void
  running: boolean
  stages: BuildStage[]
  createLandscape: () => Promise<void>
  acceptDraft: (draft: DraftResponse, name: string) => void
  intakeReport: DraftSummary | null
  substitutionMap: SubstitutionMap | null
  mapScope: MapScope
  setMapScope: (value: MapScope) => void
  scoreHistogram: ScoreHistogram | null
}

function LandscapeBuilder(props: LandscapeBuilderProps) {
  const {
    models,
    selectedModel,
    loadingModels,
    selectModel,
    parameters,
    setParameters,
    parametersLocked,
    hasStructure,
    canFitNonlinearity,
    priorId,
    setPriorId,
    oracleSeed,
    setOracleSeed,
    ensembleSize,
    setEnsembleSize,
    running,
    stages,
    createLandscape,
    acceptDraft,
    intakeReport,
    substitutionMap,
    mapScope,
    setMapScope,
    scoreHistogram,
  } = props
  const [sourceMode, setSourceMode] = useState<"preset" | "upload">("preset")
  const modelItems = Object.fromEntries(models.map((model) => [model.id, model.name]))

  return (
    <div className="studio-stack">
      <WorkflowProgress active="create" />
      <section className="page-heading">
        <div>
          <Text variant="heading2" as="h1">
            Create landscape
          </Text>
          <Text variant="secondary">
            Build a combinatorially complete fitness landscape from a reference or your DMS.
          </Text>
        </div>
      </section>

      <Grid variant="2up" gap="sm" className="builder-grid">
        <LayerCard className="panel-card">
          <div className="panel-heading">
            <div>
              <Text variant="heading3" as="h2">
                Source data
              </Text>
            </div>
            {sourceMode === "preset" && selectedModel && (
              <Badge variant={selectedModel.kind === "reference" ? "blue" : "purple"}>
                {selectedModel.kind}
              </Badge>
            )}
          </div>
          <Tabs
            size="sm"
            variant="segmented"
            value={sourceMode}
            onValueChange={(value) => setSourceMode(value as "preset" | "upload")}
            tabs={[
              { value: "preset", label: "Choose source" },
              { value: "upload", label: "Upload DMS" },
            ]}
            className="source-mode-tabs"
            listClassName="source-mode-tab-list"
          />
          {sourceMode === "preset" ? (
            <div className="source-mode-panel">
              <Select
                label="Landscape source"
                labelTooltip="A reference has every combination measured, so a synthetic landscape built on it can be scored against ground truth. A posterior is built from single mutants alone and fills the combinations from the transferred prior, which is what your own DMS gets."
                size="sm"
                value={selectedModel?.id ?? ""}
                onValueChange={(value) => value && selectModel(String(value))}
                items={modelItems}
                loading={loadingModels}
                className="full-width"
              />
              {selectedModel && (
                <div className="source-summary">
                  <div className="source-kind">
                    <Badge variant={selectedModel.kind === "reference" ? "success" : "neutral"}>
                      {selectedModel.kind === "reference"
                        ? "Combinatorially complete"
                        : "Completed from the prior"}
                    </Badge>
                    <PanelHint
                      content={
                        selectedModel.kind === "reference"
                          ? "Every combination of the mutable sites is measured. A synthetic landscape built on this source can therefore be scored against real ground truth."
                          : "A deep mutational scan measures single mutants, never the combinations. Those singles act as observations, and the bundled prior supplies the pairwise structure learnt from other proteins. Conditioning one on the other gives a distribution over landscapes rather than one answer, which is why the panels show shaded bands. The oracle seed fixes a single draw so every query stays reproducible."
                      }
                    />
                  </div>
                  <Text size="sm">{selectedModel.description}</Text>
                  <dl className="compact-facts">
                    <div>
                      <dt>Mutable sites</dt>
                      <dd>{selectedModel.positions.length}</dd>
                    </div>
                    <div>
                      <dt>Parent</dt>
                      <dd className="mono-value">{selectedModel.wtCompact}</dd>
                    </div>
                    {selectedModel.kind === "posterior" ? (
                      <div>
                        <dt>DMS anchors</dt>
                        <dd>{selectedModel.sourceSummary?.observationCount ?? "Unknown"}</dd>
                      </div>
                    ) : (
                      <div>
                        <dt>Structure</dt>
                        <dd>
                          {selectedModel.sourceSummary?.structureMethod === "esmfold"
                            ? "ESMFold"
                            : hasStructure
                              ? "Available"
                              : "None"}
                        </dd>
                      </div>
                    )}
                  </dl>
                </div>
              )}
            </div>
          ) : (
            <CustomUploadPanel
              onCreated={(created, name) => {
                acceptDraft(created, name)
                setSourceMode("preset")
              }}
            />
          )}
        </LayerCard>

        {selectedModel?.kind === "posterior" ? (
          <PosteriorParameterPanel
            priorId={priorId}
            setPriorId={setPriorId}
            oracleSeed={oracleSeed}
            setOracleSeed={setOracleSeed}
            ensembleSize={ensembleSize}
            setEnsembleSize={setEnsembleSize}
            locked={!selectedModel.createdAt ? false : !selectedModel.id.startsWith("draft_")}
          />
        ) : (
          <ParameterPanel
            value={parameters}
            onChange={setParameters}
            locked={parametersLocked}
            hasSource={selectedModel !== null}
            hasStructure={hasStructure}
            canFitNonlinearity={canFitNonlinearity}
          />
        )}
      </Grid>

      {intakeReport && <IntakeReport summary={intakeReport} />}
      {!selectedModel && sourceMode === "preset" && (
        <LayerCard className="compact-empty panel-card">
          <Empty
            size="base"
            icon={<Flask size={36} />}
            title="No source data"
            description="Choose a source above or upload your own DMS."
          />
        </LayerCard>
      )}
      {substitutionMap && (
        <SubstitutionMapPanel map={substitutionMap} scope={mapScope} onScopeChange={setMapScope} />
      )}
      {scoreHistogram && <ScoreHistogramPanel histogram={scoreHistogram} />}
      {props.structureSource && (
        <StructureViewerPanel
          key={`${props.structureSource.kind}:${props.structureSource.id}`}
          source={props.structureSource}
        />
      )}
      <BuildProgress stages={stages} />

      <div className="builder-action-bar">
        <div>
          <Text bold size="sm">
            {sourceMode === "upload"
              ? "Upload source files"
              : (selectedModel?.name ?? "Choose a source")}
          </Text>
          <Text variant="secondary" size="xs">
            {sourceMode === "upload"
              ? "Use the uploaded data before creating the landscape."
              : "Creation uses the current parameters and opens the landscape explorer."}
          </Text>
        </div>
        <Button
          size="sm"
          variant="primary"
          icon={<Play weight="fill" />}
          loading={running}
          disabled={sourceMode === "upload" || !selectedModel || running}
          onClick={() => void createLandscape()}
        >
          Create landscape
        </Button>
      </div>
    </div>
  )
}

interface SamplingProps {
  selectedModel: ModelSummary | null
  sampleSize: number
  sampleLimit: number
  setSampleSize: (value: number) => void
  maxDepth: number
  changeMaxDepth: (value: number) => void
  depthWeights: Record<number, number>
  setDepthWeights: (next: Record<number, number>) => void
  running: boolean
  stages: BuildStage[]
  applyPreview: () => Promise<boolean>
}

function IntakeReport({ summary }: { summary: DraftSummary }) {
  const skipped = Object.entries(summary.skippedRows).sort((left, right) => right[1] - left[1])
  if (skipped.length === 0 && !summary.wildTypeScoreAssumed) return null
  const total = skipped.reduce((sum, [, count]) => sum + count, 0)
  const title = total > 0 ? `${total} rows dropped` : "Parent score assumed"

  return (
    <Banner variant="alert" title={title} className="intake-report">
      <div className="intake-report-body">
        {skipped.length > 0 && (
          <ul className="intake-reasons">
            {skipped.map(([reason, count]) => (
              <li key={reason}>
                <span>{reason}</span>
                <Badge variant="secondary">{count}</Badge>
              </li>
            ))}
          </ul>
        )}
        {summary.wildTypeScoreAssumed && (
          <Text size="sm">
            No wild-type or synonymous row was found, so the parent score was taken as 0. Set a
            wild-type score on the upload panel if this score set puts the parent elsewhere.
          </Text>
        )}
      </div>
    </Banner>
  )
}

function SubstitutionMapPanel({
  map,
  scope,
  onScopeChange,
}: {
  map: SubstitutionMap
  scope: MapScope
  onScopeChange: (value: MapScope) => void
}) {
  const total = map.measuredCount + map.imputedCount
  const caveats = [
    map.imputedCount > 0
      ? `${map.imputedCount.toLocaleString()} of ${total.toLocaleString()} cells are imputed by position mean.`
      : null,
    scope === "sequence"
      ? `${map.assayedCount.toLocaleString()} of ${map.sequenceLength.toLocaleString()} residues were assayed.`
      : null,
    map.truncated
      ? `Showing the first ${map.positions.length} of ${map.totalPositions.toLocaleString()} sites.`
      : null,
  ].filter(Boolean)

  return (
    <LayerCard className="panel-card chart-card substitution-map-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Single-mutant effects
          </Text>
          {caveats.length > 0 && (
            <Text variant="secondary" size="sm">
              {caveats.join(" ")}
            </Text>
          )}
        </div>
        <div className="panel-heading-controls">
          <Tabs
            size="sm"
            variant="segmented"
            value={scope}
            onValueChange={(value) => onScopeChange(value as MapScope)}
            tabs={[
              { value: "assayed", label: "Assayed sites" },
              { value: "sequence", label: "Whole sequence" },
            ]}
          />
          <Badge variant="neutral">{map.measuredCount.toLocaleString()} measured</Badge>
        </div>
      </div>
      <Chart echarts={echarts} options={substitutionMapOptions(map)} height={430} />
      <ul className="substitution-map-legend">
        <li>
          <span className="swatch measured" />
          Measured
        </li>
        {map.imputedCount > 0 && (
          <li>
            <span className="swatch imputed" aria-hidden="true">
              ×
            </span>
            Imputed from the position mean
          </li>
        )}
        <li>
          <span className="swatch parent" />
          Parent residue
        </li>
      </ul>
    </LayerCard>
  )
}

function ScoreHistogramPanel({ histogram }: { histogram: ScoreHistogram }) {
  return (
    <LayerCard className="panel-card chart-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Functional score distribution
          </Text>
          <Text variant="secondary" size="sm">
            Scores run from {histogram.min.toFixed(2)} to {histogram.max.toFixed(2)} with a median
            of {histogram.median.toFixed(2)}.
          </Text>
        </div>
        <Badge variant="neutral">{histogram.singleCount.toLocaleString()} singles</Badge>
      </div>
      <Chart echarts={echarts} options={scoreHistogramOptions(histogram)} height={280} />
    </LayerCard>
  )
}

function LandscapeExplorer({
  selectedModel,
  parameters,
  priorId,
  oracleSeed,
  ensembleSize,
  preview,
  posteriorPreview,
  draft,
  draftName,
  publish,
  publishedExamples,
  onBack,
  ...sampling
}: SamplingProps & {
  parameters: ModelParameters
  priorId: string
  oracleSeed: number
  ensembleSize: number
  preview: PreviewResponse | null
  posteriorPreview: PosteriorPreviewResponse | null
  draft: DraftResponse | null
  draftName: string
  publish: (name: string) => Promise<void>
  publishedExamples: CodeExamples | null
  onBack: () => void
}) {
  return (
    <div className="studio-stack">
      <WorkflowProgress active="explore" />
      <section className="page-heading">
        <div>
          <Text variant="heading2" as="h1">
            Explore {selectedModel?.name ?? "landscape"}
          </Text>
          <Text variant="secondary">
            Sample variants, inspect fitness distributions, and query this landscape.
          </Text>
        </div>
        <Button size="sm" variant="secondary" icon={<ArrowLeft />} onClick={onBack}>
          Back to creation
        </Button>
      </section>

      <ExactQueryPanel
        key={`${selectedModel?.id ?? ""}:${priorId}:${oracleSeed}:${ensembleSize}:${Object.values(parameters).join(":")}`}
        selectedModel={selectedModel}
        draft={draft}
        parameters={parameters}
        priorId={priorId}
        oracleSeed={oracleSeed}
        ensembleSize={ensembleSize}
      />
      <SamplePanel
        {...sampling}
        selectedModel={selectedModel}
        hasPreview={preview !== null || posteriorPreview !== null}
      />

      {posteriorPreview ? (
        <PosteriorPreviewResults
          preview={posteriorPreview}
          draft={draft}
          draftName={draftName}
          publish={publish}
          examples={publishedExamples}
        />
      ) : preview ? (
        <PreviewResults
          preview={preview}
          draft={draft}
          draftName={draftName}
          publish={publish}
          examples={publishedExamples}
        />
      ) : (
        <LayerCard className="compact-empty panel-card">
          <Empty
            size="base"
            icon={<Flask size={36} />}
            title="Sample this landscape"
            description="Choose the sample size and mutation-depth mix above."
          />
        </LayerCard>
      )}
    </div>
  )
}

function mutationExample(model: ModelSummary | null): string {
  if (!model) return "V39A+D40G"
  return model.positions
    .slice(0, 2)
    .map((position, index) => {
      const wildType = model.wtCompact[index]
      const mutant = wildType === "A" ? "G" : "A"
      return `${wildType}${position}${mutant}`
    })
    .join("+")
}

const AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
const TOKEN_SEPARATOR = /[+:,\s]+/

function CompactSequence({ sequence, parent }: { sequence: string; parent: string }) {
  return (
    <span className="mono-value compact-sequence">
      {sequence.split("").map((residue, index) => {
        const substituted = residue !== parent[index]
        return (
          <span
            // Position is the identity here. Residues repeat, so the character cannot key it.
            // biome-ignore lint/suspicious/noArrayIndexKey: index is the site index
            key={index}
            className={substituted ? "substituted" : undefined}
            title={substituted ? `${parent[index]} replaced by ${residue}` : undefined}
          >
            {residue}
          </span>
        )
      })}
    </span>
  )
}

interface VariantSuggestions {
  prefix: string
  kind: "site" | "substitution"
  options: string[]
}

// Completes the token under the caret only. Splitting on the separators the parser accepts
// keeps the suggestion aligned with whatever the user is part-way through typing, so a
// finished substitution earlier in the string is never rewritten.
function variantSuggestions(model: ModelSummary | null, text: string): VariantSuggestions | null {
  if (!model) return null
  const parts = text.split(TOKEN_SEPARATOR)
  const token = (parts.at(-1) ?? "").toUpperCase()
  const prefix = text.slice(0, text.length - (parts.at(-1) ?? "").length)
  const shape = token.match(/^([A-Z]?)(\d*)([A-Z]?)$/)
  if (!shape) return null
  const [, parent, digits, mutant] = shape

  const taken = new Set(
    parts
      .slice(0, -1)
      .map((part) => part.toUpperCase().replace(/[^0-9]/g, ""))
      .filter(Boolean),
  )
  const sites = model.positions
    .map((position, index) => ({ position, parent: model.wtCompact[index] ?? "" }))
    .filter((site) => !taken.has(String(site.position)))
    .filter((site) => !parent || site.parent === parent)
    .filter((site) => String(site.position).startsWith(digits))

  const exact = sites.find((site) => String(site.position) === digits)
  if (digits && exact && (parent === exact.parent || parent === "")) {
    const options = AMINO_ACIDS.split("")
      .filter((amino) => amino !== exact.parent)
      .filter((amino) => !mutant || amino.startsWith(mutant))
      .map((amino) => `${exact.parent}${exact.position}${amino}`)
    if (options.length > 0) return { prefix, kind: "substitution", options }
  }
  const options = sites.map((site) => `${site.parent}${site.position}`)
  return options.length > 0 ? { prefix, kind: "site", options } : null
}

function ExactQueryPanel({
  selectedModel,
  draft,
  parameters,
  priorId,
  oracleSeed,
  ensembleSize,
}: {
  selectedModel: ModelSummary | null
  draft: DraftResponse | null
  parameters: ModelParameters
  priorId: string
  oracleSeed: number
  ensembleSize: number
}) {
  const [mutation, setMutation] = useState("")
  const [result, setResult] = useState<EvaluationResult | null>(null)
  const [posteriorResults, setPosteriorResults] = useState<PosteriorQueryResult[]>([])
  const [querying, setQuerying] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)

  const latestPosterior = posteriorResults.at(-1)
  const suggestions = useMemo(
    () => variantSuggestions(selectedModel, mutation),
    [selectedModel, mutation],
  )

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!selectedModel || !mutation.trim()) return
    setQuerying(true)
    setQueryError(null)
    try {
      const source = sourceFromModel(selectedModel, draft?.id)
      const label = mutation.trim().toUpperCase()
      if (selectedModel.kind === "posterior") {
        const response = await queryPosterior(source, priorId, oracleSeed, ensembleSize, label)
        const next = response.results[0]
        setPosteriorResults((current) =>
          [...current.filter((item) => item.mutationLabel !== next.mutationLabel), next].slice(-10),
        )
        setResult(null)
      } else {
        setResult(await evaluateLandscape(source, parameters, label))
      }
    } catch (cause) {
      setResult(null)
      setQueryError(cause instanceof Error ? cause.message : "Could not score this mutation.")
    } finally {
      setQuerying(false)
    }
  }

  return (
    <LayerCard className="panel-card query-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Query an exact variant
          </Text>
        </div>
      </div>
      <form className="query-form" onSubmit={(event) => void submit(event)}>
        <Input
          size="sm"
          label="Mutation set"
          labelTooltip={`Parent residue, position, then the new residue, for example ${mutationExample(selectedModel)}. Separate substitutions with + or a colon. Lowercase works. Type WT for the parent sequence.`}
          placeholder={mutationExample(selectedModel)}
          value={mutation}
          onChange={(event) => setMutation(event.currentTarget.value)}
          error={queryError ?? undefined}
        />
        <Button
          size="sm"
          type="submit"
          variant="primary"
          loading={querying}
          disabled={!selectedModel || !mutation.trim() || querying}
        >
          Score variant
        </Button>
        {result && (
          <div className="query-result" aria-live="polite">
            <span>{result.mutationLabel}</span>
            <strong>{result.fitness.toFixed(4)}</strong>
            <small>synthetic fitness</small>
          </div>
        )}
      </form>
      {suggestions && (
        <div className="variant-suggestions">
          <Text variant="secondary" size="xs">
            {suggestions.kind === "site" ? "Mutable sites" : "Replacement residues"}
          </Text>
          <div className="variant-suggestion-row">
            {suggestions.options.map((option) => (
              <button
                key={option}
                type="button"
                className="variant-chip"
                onClick={() =>
                  setMutation(
                    suggestions.kind === "site"
                      ? `${suggestions.prefix}${option}`
                      : `${suggestions.prefix}${option}+`,
                  )
                }
              >
                {suggestions.kind === "site" ? option : option.slice(-1)}
              </button>
            ))}
          </div>
        </div>
      )}
      {latestPosterior && (
        <div className="posterior-query-results" aria-live="polite">
          <div className="panel-heading">
            <Badge variant={latestPosterior.observed ? "green" : "purple"}>
              {latestPosterior.observed ? "Measured observation" : "Synthetic oracle value"}
            </Badge>
            <Badge variant="neutral">{posteriorResults.length} unique session queries</Badge>
          </div>
          <div className="posterior-metric-grid">
            <div>
              <span>Fixed oracle</span>
              <strong>{latestPosterior.fixedOracleFitness.toFixed(4)}</strong>
            </div>
            <div>
              <span>Ensemble mean</span>
              <strong>{latestPosterior.posteriorMean.toFixed(4)}</strong>
            </div>
            <div>
              <span>Oracle spread</span>
              <strong>{latestPosterior.posteriorStd.toFixed(4)}</strong>
            </div>
            <div>
              <span>80% interval</span>
              <strong>
                {latestPosterior.intervals["80"].lower.toFixed(3)} to{" "}
                {latestPosterior.intervals["80"].upper.toFixed(3)}
              </strong>
            </div>
          </div>
          <Grid variant="2up" gap="sm">
            <div className="embedded-chart">
              <Text bold size="sm">
                Queried variants
              </Text>
              <PosteriorQueryPlot results={posteriorResults} />
            </div>
            <div className="embedded-chart">
              <Text bold size="sm">
                Latest oracle distribution
              </Text>
              <PosteriorDistributionPlot result={latestPosterior} />
            </div>
          </Grid>
        </div>
      )}
    </LayerCard>
  )
}

function PosteriorQueryPlot({ results }: { results: PosteriorQueryResult[] }) {
  const values = results.flatMap((result) => [
    result.intervals["80"].lower,
    result.intervals["80"].upper,
    result.additiveFitness,
    result.fixedOracleFitness,
    result.posteriorMean,
  ])
  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const range = maximum - minimum || 1
  const position = (value: number) => `${((value - minimum) / range) * 100}%`
  return (
    <div className="interval-plot" role="img" aria-label="Oracle ranges for queried variants">
      {results.map((result) => (
        <div className="interval-row" key={result.mutationLabel}>
          <span className="mono-value">{result.mutationLabel}</span>
          <div className="interval-track">
            <i
              className="interval-band"
              style={{
                left: position(result.intervals["80"].lower),
                width: `${((result.intervals["80"].upper - result.intervals["80"].lower) / range) * 100}%`,
              }}
            />
            <i
              className="interval-point additive"
              style={{ left: position(result.additiveFitness) }}
            />
            <i className="interval-point mean" style={{ left: position(result.posteriorMean) }} />
            <i
              className="interval-point fixed"
              style={{ left: position(result.fixedOracleFitness) }}
            />
          </div>
        </div>
      ))}
      <div className="interval-axis">
        <span>{minimum.toFixed(2)}</span>
        <span>{maximum.toFixed(2)}</span>
      </div>
    </div>
  )
}

function PosteriorDistributionPlot({ result }: { result: PosteriorQueryResult }) {
  const minimum = Math.min(...result.samples)
  const maximum = Math.max(...result.samples)
  const bins = Array.from({ length: 14 }, (_, index) => ({ index, count: 0 }))
  const width = (maximum - minimum) / bins.length || 1
  for (const value of result.samples) {
    const index = Math.min(bins.length - 1, Math.floor((value - minimum) / width))
    bins[index].count += 1
  }
  const peak = Math.max(...bins.map((bin) => bin.count), 1)
  return (
    <div
      className="histogram"
      role="img"
      aria-label={`Oracle distribution for ${result.mutationLabel}`}
    >
      <div className="histogram-bars">
        {bins.map((bin) => (
          <i
            key={(minimum + bin.index * width).toFixed(8)}
            style={{ height: `${(bin.count / peak) * 100}%` }}
          />
        ))}
      </div>
      <div className="histogram-axis">
        <span>{minimum.toFixed(2)}</span>
        <span>{maximum.toFixed(2)}</span>
      </div>
    </div>
  )
}

function NumberStepper({
  label,
  tooltip,
  value,
  min,
  max,
  onChange,
}: {
  label: string
  tooltip: string
  value: number
  min: number
  max: number
  onChange: (next: number) => void
}) {
  // The field holds a string so an empty box stays empty while typing. Committing
  // an unparseable value would otherwise snap the caret back to the clamped number.
  const [text, setText] = useState(String(value))
  useEffect(() => setText(String(value)), [value])

  function commit(raw: string) {
    const parsed = Number(raw)
    if (!Number.isFinite(parsed)) {
      setText(String(value))
      return
    }
    onChange(Math.max(min, Math.min(max, Math.round(parsed))))
  }

  return (
    <div className="number-stepper">
      <Input
        size="sm"
        label={label}
        labelTooltip={tooltip}
        type="number"
        min={min}
        max={max}
        value={text}
        onChange={(event) => setText(event.currentTarget.value)}
        onBlur={(event) => commit(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit(event.currentTarget.value)
        }}
      />
      <div className="number-stepper-buttons">
        <Button
          size="sm"
          variant="secondary"
          aria-label={`Decrease ${label.toLowerCase()}`}
          icon={<Minus />}
          disabled={value <= min}
          onClick={() => onChange(Math.max(min, value - 1))}
        />
        <Button
          size="sm"
          variant="secondary"
          aria-label={`Increase ${label.toLowerCase()}`}
          icon={<Plus />}
          disabled={value >= max}
          onClick={() => onChange(Math.min(max, value + 1))}
        />
      </div>
    </div>
  )
}

function SamplePanel({
  selectedModel,
  sampleSize,
  sampleLimit,
  setSampleSize,
  maxDepth,
  changeMaxDepth,
  depthWeights,
  setDepthWeights,
  running,
  stages,
  applyPreview,
  hasPreview,
}: SamplingProps & { hasPreview: boolean }) {
  const depthCeiling = selectedModel?.positions.length ?? 1
  return (
    <LayerCard className="panel-card sample-controls">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Sample the landscape
          </Text>
        </div>
      </div>
      <div className="sample-control-grid">
        <Input
          size="sm"
          label="Variants"
          labelTooltip={`Unique genotypes to draw, up to ${sampleLimit.toLocaleString()} for this landscape.`}
          type="number"
          min={1}
          max={sampleLimit}
          value={sampleSize}
          onChange={(event) => setSampleSize(Number(event.currentTarget.value))}
        />
        <NumberStepper
          label="Maximum depth"
          tooltip={`Most substitutions from the parent in one variant. Depth 1 is a single mutant, depth 2 a double. This landscape has ${depthCeiling} mutable sites.`}
          value={maxDepth}
          min={1}
          max={depthCeiling}
          onChange={changeMaxDepth}
        />
        {Object.keys(depthWeights).map((rawDepth) => {
          const depth = Number(rawDepth)
          return (
            <Input
              size="sm"
              key={depth}
              label={`Depth ${depth} share`}
              labelTooltip={`Relative share of the sample given to ${depth}-substitution variants. Set 0 to leave depth ${depth} out.`}
              type="number"
              min={0}
              step={0.1}
              value={depthWeights[depth]}
              onChange={(event) =>
                setDepthWeights({
                  ...depthWeights,
                  [depth]: Number(event.currentTarget.value),
                })
              }
            />
          )
        })}
        <Button
          size="sm"
          variant="primary"
          icon={<Play weight="fill" />}
          loading={running}
          disabled={!selectedModel || running}
          onClick={() => void applyPreview()}
          className="sample-button"
        >
          {hasPreview ? "Update sample" : "Sample landscape"}
        </Button>
      </div>
      <BuildProgress stages={stages} />
    </LayerCard>
  )
}

function WorkflowProgress({ active }: { active: "create" | "explore" }) {
  return (
    <nav className="workflow-progress" aria-label="Landscape workflow">
      <div className={active === "create" ? "workflow-stage active" : "workflow-stage done"}>
        <span>1</span>
        Create landscape
      </div>
      <div className="workflow-connector" />
      <div className={active === "explore" ? "workflow-stage active" : "workflow-stage"}>
        <span>2</span>
        Explore landscape
      </div>
    </nav>
  )
}

function PosteriorParameterPanel({
  priorId,
  setPriorId,
  oracleSeed,
  setOracleSeed,
  ensembleSize,
  setEnsembleSize,
  locked,
}: {
  priorId: string
  setPriorId: (value: string) => void
  oracleSeed: number
  setOracleSeed: (value: number) => void
  ensembleSize: number
  setEnsembleSize: (value: number) => void
  locked: boolean
}) {
  return (
    <LayerCard className="panel-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Oracle ensemble
          </Text>
        </div>
        {locked && <Badge variant="neutral">Fixed oracle</Badge>}
      </div>
      <div className="field-grid">
        <Select
          size="sm"
          label="Readout calibration"
          labelTooltip="Which empirical prior supplies the typical assay range used by the saturating readout."
          value={priorId}
          items={{ "combinatorial-v2": "Combinatorial v2" }}
          disabled={locked}
          onValueChange={(value) => value && setPriorId(String(value))}
        />
        <Input
          size="sm"
          label="Oracle seed"
          labelTooltip="Picks one draw from the posterior to act as the benchmark oracle. Every query against this landscape answers from that single draw, so results stay comparable across runs."
          type="number"
          step={1}
          value={oracleSeed}
          disabled={locked}
          onChange={(event) => setOracleSeed(Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Oracle draws"
          labelTooltip="How many posterior draws the uncertainty ranges are estimated from. This affects the shaded intervals only, never the oracle itself. More draws give smoother ranges and take longer."
          type="number"
          min={2}
          max={200}
          step={1}
          value={ensembleSize}
          onChange={(event) => setEnsembleSize(Number(event.currentTarget.value))}
        />
      </div>
      <Banner
        size="sm"
        variant="secondary"
        title="Empirical readout prior"
        description="Measured combinatorial landscapes supply the typical assay range. Target singles set the local fitness scale."
      />
    </LayerCard>
  )
}

interface ParameterPanelProps {
  value: ModelParameters
  onChange: (value: ModelParameters) => void
  locked: boolean
  hasSource: boolean
  hasStructure: boolean
  canFitNonlinearity: boolean
}

function ParameterPanel({
  value,
  onChange,
  locked,
  hasSource,
  hasStructure,
  canFitNonlinearity,
}: ParameterPanelProps) {
  function set<K extends keyof ModelParameters>(key: K, next: ModelParameters[K]) {
    onChange({ ...value, [key]: next })
  }
  return (
    <LayerCard className="panel-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Landscape behaviour
          </Text>
        </div>
        {locked && <Badge variant="neutral">Fixed</Badge>}
      </div>
      <div className="field-grid">
        <Input
          size="sm"
          label="Ruggedness"
          labelTooltip="Standard deviation of the Gaussian noise added before the fitness readout. The draw is pinned to the genotype, so the same variant always returns the same number. Raising it creates more local optima."
          type="number"
          min={0}
          max={5}
          step={0.05}
          value={value.noiseSigma}
          disabled={locked}
          onChange={(event) => set("noiseSigma", Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Fitness headroom"
          labelTooltip="Upper asymptote of the global-epistasis readout, in units of wild-type fitness. It caps how far stacked beneficial mutations can climb, which is what produces diminishing returns."
          type="number"
          min={1}
          max={10}
          step={0.1}
          value={value.combinatorialUpside}
          disabled={locked || !canFitNonlinearity}
          onChange={(event) => set("combinatorialUpside", Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Pairwise sigma"
          labelTooltip="Standard deviation of the random coupling drawn for each contacting pair of sites. Zero leaves only the couplings implied by the measured data. Raising it above zero worsened the headline metrics on both reference proteins."
          type="number"
          min={0}
          max={5}
          step={0.05}
          value={value.pairwiseSigma}
          disabled={locked || !hasStructure}
          onChange={(event) => set("pairwiseSigma", Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Pairwise alpha"
          labelTooltip="Multiplier on the whole pairwise term. Only pairs of sites in structural contact contribute, so this scales specific epistasis without touching the additive part."
          type="number"
          min={0}
          max={10}
          step={0.1}
          value={value.alpha}
          disabled={locked || !hasStructure}
          onChange={(event) => set("alpha", Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Seed"
          labelTooltip="Seeds the noise draws and the random couplings. The same seed and parameters rebuild an identical landscape."
          type="number"
          step={1}
          value={value.seed}
          disabled={locked}
          onChange={(event) => set("seed", Number(event.currentTarget.value))}
        />
      </div>
      {hasSource && !hasStructure && (
        <Banner
          size="sm"
          variant="secondary"
          title="No structural contacts"
          description="Pairwise sigma and alpha stay inactive for this model."
        />
      )}
      {hasSource && !canFitNonlinearity && (
        <Banner
          size="sm"
          variant="alert"
          title="Identity fitness readout"
          description="The measured single effects do not cross zero, so fitness headroom is inactive."
        />
      )}
    </LayerCard>
  )
}

interface BuildStage {
  name: string
  startedAt: number
  endedAt: number | null
}

function formatElapsed(milliseconds: number): string {
  return `${(Math.max(milliseconds, 0) / 1000).toFixed(1)}s`
}

function BuildProgress({ stages }: { stages: BuildStage[] }) {
  const open = stages.some((stage) => stage.endedAt === null)
  const [now, setNow] = useState(() => performance.now())

  useEffect(() => {
    if (!open) return
    // The reading held from the last run is stale, so take a fresh one before the first tick.
    setNow(performance.now())
    const timer = window.setInterval(() => setNow(performance.now()), 100)
    return () => window.clearInterval(timer)
  }, [open])

  if (stages.length === 0) return null
  const first = stages[0]
  const last = stages[stages.length - 1]
  const total = (last.endedAt ?? now) - first.startedAt

  return (
    <section className="build-progress">
      <ul className="build-stage-list">
        {stages.map((stage) => (
          <li
            className={stage.endedAt === null ? "build-stage running" : "build-stage done"}
            key={stage.name}
          >
            <span className="build-stage-dot" />
            <Text size="sm">{stage.name}</Text>
            <Text variant="secondary" size="xs">
              {formatElapsed((stage.endedAt ?? now) - stage.startedAt)}
            </Text>
          </li>
        ))}
      </ul>
      <Badge variant={open ? "purple" : "neutral"}>{formatElapsed(total)} total</Badge>
    </section>
  )
}

function PanelHint({ content }: { content: string }) {
  return (
    <Tooltip content={content} delay={150}>
      <button type="button" className="panel-hint" aria-label={content}>
        <Info size={15} weight="bold" />
      </button>
    </Tooltip>
  )
}

function useNarrowViewport(query = "(max-width: 760px)"): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && (window.matchMedia?.(query).matches ?? false),
  )
  useEffect(() => {
    const media = window.matchMedia?.(query)
    if (!media) return
    const update = () => setNarrow(media.matches)
    update()
    media.addEventListener("change", update)
    return () => media.removeEventListener("change", update)
  }, [query])
  return narrow
}

function LandscapeMapPanel({ graph, heading }: { graph: LandscapeGraph; heading: string }) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const narrow = useNarrowViewport()
  const selected = graph.nodes.find((node) => node.id === selectedId) ?? null
  // A selection that survives into a new sample is dropped by the lookup above, so the panel
  // needs no reset effect.
  const reached = selected ? (reachableFrom(graph, selected.id)?.size ?? 1) - 1 : 0
  const deepest = Math.max(...graph.nodes.map((node) => node.depth), 0)

  return (
    <LayerCard className="panel-card chart-card landscape-map-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            {heading}
          </Text>
          <Text variant="secondary" size="sm">
            Click a marker to trace its mutation path. Scroll to zoom.
          </Text>
        </div>
        {selected ? (
          <div className="map-selection">
            <Badge variant="purple">
              {selected.id === "wt" ? "Parent" : selected.variant}, {reached} connected
            </Badge>
            <Button size="sm" variant="secondary" onClick={() => setSelectedId(null)}>
              Clear
            </Button>
          </div>
        ) : (
          <div className="map-selection">
            <Badge variant="neutral">{graph.nodes.length} variants</Badge>
            <Badge variant="neutral">up to {deepest} mutations</Badge>
          </div>
        )}
      </div>
      <Chart
        echarts={echarts}
        options={landscapeGraphOptions(graph, selectedId)}
        height={narrow ? 300 : 430}
        onEvents={{
          click: (params) => {
            if (params.dataType !== "node") return
            const id = (params.data as { id?: string })?.id
            if (id) setSelectedId((current) => (current === id ? null : id))
          },
        }}
      />
    </LayerCard>
  )
}

function NoticeCard({ heading, children }: { heading: string; children: ReactNode }) {
  return (
    <LayerCard className="panel-card notice-card">
      <div className="panel-heading">
        <Text variant="heading3" as="h2">
          {heading}
        </Text>
      </div>
      {children}
    </LayerCard>
  )
}

function PreviewResults({
  preview,
  draft,
  draftName,
  publish,
  examples,
}: {
  preview: PreviewResponse
  draft: DraftResponse | null
  draftName: string
  publish: (name: string) => Promise<void>
  examples: CodeExamples | null
}) {
  return (
    <div className="results-stack">
      <LandscapeMapPanel graph={preview.sample.graph} heading="Interactive landscape map" />

      <Grid variant="2up" gap="sm">
        <LayerCard className="panel-card chart-card">
          <div className="panel-heading">
            <div>
              <Text variant="heading3" as="h2">
                Fitness by mutation depth
              </Text>
              <Text variant="secondary" size="sm">
                Median, middle half, and 5 to 95% range.
              </Text>
            </div>
            <Badge variant={preview.cached ? "green" : "neutral"} appearance="dot">
              {preview.cached ? "Cached" : "Calculated"}
            </Badge>
          </div>
          <Chart echarts={echarts} options={fitnessProfileOptions(preview.profiles)} height={320} />
        </LayerCard>

        {preview.walsh && (
          <LayerCard className="panel-card chart-card">
            <div className="panel-heading">
              <Text variant="heading3" as="h2">
                Interaction order
              </Text>
              <PanelHint content="Fraction of fitness variance carried by each Walsh order. Order 1 is additive, order 2 pairwise, and so on. The measured bars need every combination of the mutated sites, so this panel only appears for a combinatorially complete reference. An uploaded DMS has no ground truth to decompose and gets no bars." />
            </div>
            <Chart echarts={echarts} options={walshOptions(preview.walsh)} height={250} />
          </LayerCard>
        )}

        {preview.walsh && <MetricTable rows={preview.metrics} />}

        {preview.notices.length > 0 && (
          <NoticeCard heading="Notices">
            {preview.notices.map((notice) => (
              <Banner key={notice} size="sm" variant="secondary" description={notice} />
            ))}
          </NoticeCard>
        )}
      </Grid>

      <SampleTable preview={preview} />

      {draft && <PublishPanel defaultName={draftName} onPublish={publish} />}
      {examples && <CodePanel examples={examples} />}
    </div>
  )
}

function PosteriorPreviewResults({
  preview,
  draft,
  draftName,
  publish,
  examples,
}: {
  preview: PosteriorPreviewResponse
  draft: DraftResponse | null
  draftName: string
  publish: (name: string) => Promise<void>
  examples: CodeExamples | null
}) {
  return (
    <div className="results-stack">
      <LandscapeMapPanel graph={preview.sample.graph} heading="Synthetic landscape map" />

      <Grid variant="2up" gap="sm">
        <LayerCard className="panel-card chart-card">
          <div className="panel-heading">
            <div>
              <Text variant="heading3" as="h2">
                Fitness variation by mutation depth
              </Text>
              <Text variant="secondary" size="sm">
                Bands pool fixed oracle draws. Lines compare one oracle with the additive model.
              </Text>
            </div>
            <Badge variant="purple">{preview.settings.ensembleSize} draws</Badge>
          </div>
          <Chart
            echarts={echarts}
            options={posteriorProfileOptions(preview.profile)}
            height={340}
          />
        </LayerCard>

        <NoticeCard heading={preview.prior.name}>
          <Banner
            size="sm"
            variant="secondary"
            description={`The readout calibration uses ${preview.prior.nLandscapes} landscapes across ${preview.prior.nProteins} proteins. Each draw adds stable random pair interactions at coupling ${preview.settings.coupling}.`}
          />
          {preview.notices.map((notice) => (
            <Banner key={notice} size="sm" variant="secondary" description={notice} />
          ))}
        </NoticeCard>
      </Grid>

      <PosteriorSampleTable preview={preview} />
      {draft && <PublishPanel defaultName={draftName} onPublish={publish} />}
      {examples && <CodePanel examples={examples} />}
    </div>
  )
}

function PosteriorSampleTable({ preview }: { preview: PosteriorPreviewResponse }) {
  return (
    <LayerCard className="panel-card sample-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Oracle sample explorer
          </Text>
          <Text variant="secondary" size="sm">
            Showing {preview.sample.returnedRows} rows from {preview.sample.requested} sampled
            variants.
          </Text>
        </div>
      </div>
      <div className="table-scroll sample-table-scroll">
        <Table layout="auto">
          <Table.Header>
            <Table.Row>
              <Table.Head>Variant</Table.Head>
              <Table.Head>Depth</Table.Head>
              <Table.Head>Fixed</Table.Head>
              <Table.Head>Ensemble mean</Table.Head>
              <Table.Head>Spread</Table.Head>
              <Table.Head>80% interval</Table.Head>
              <Table.Head>Additive</Table.Head>
              <Table.Head>Sequence</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {preview.sample.rows.map((row) => (
              <Table.Row key={row.variant}>
                <Table.Cell>
                  <span className="mono-value">{row.variant}</span>
                </Table.Cell>
                <Table.Cell>{row.depth}</Table.Cell>
                <Table.Cell>{row.fixedOracleFitness.toFixed(3)}</Table.Cell>
                <Table.Cell>{row.posteriorMean.toFixed(3)}</Table.Cell>
                <Table.Cell>{row.posteriorStd.toFixed(3)}</Table.Cell>
                <Table.Cell>
                  {row.lower80.toFixed(2)} to {row.upper80.toFixed(2)}
                </Table.Cell>
                <Table.Cell>{row.additiveFitness.toFixed(3)}</Table.Cell>
                <Table.Cell>
                  <CompactSequence sequence={row.sequence} parent={preview.source.wtCompact} />
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </LayerCard>
  )
}

function MetricTable({ rows }: { rows: PreviewResponse["metrics"] }) {
  return (
    <LayerCard className="panel-card metric-card">
      <Text variant="heading3" as="h2">
        Validation metrics
      </Text>
      <Text variant="secondary" size="sm">
        Lower is better for W1 and L1. Other rows describe topology.
      </Text>
      <div className="table-scroll">
        <Table>
          <Table.Header>
            <Table.Row>
              <Table.Head>Metric</Table.Head>
              <Table.Head>Measured</Table.Head>
              <Table.Head>Synthetic</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {rows.map((row) => (
              <Table.Row key={row.name}>
                <Table.Cell>{row.name}</Table.Cell>
                <Table.Cell>{formatNumber(row.real, row.name)}</Table.Cell>
                <Table.Cell>{formatNumber(row.synthetic, row.name)}</Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </LayerCard>
  )
}

function SampleTable({ preview }: { preview: PreviewResponse }) {
  return (
    <LayerCard className="panel-card sample-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Sample explorer
          </Text>
          <Text variant="secondary" size="sm">
            Showing {preview.sample.returnedRows} representative rows from a unique sample of{" "}
            {preview.sample.requested.toLocaleString()} variants.
          </Text>
        </div>
      </div>
      <div className="table-scroll sample-table-scroll">
        <Table layout="auto">
          <Table.Header>
            <Table.Row>
              <Table.Head>Variant</Table.Head>
              <Table.Head>Depth</Table.Head>
              <Table.Head>Synthetic</Table.Head>
              {preview.comparisonAvailable && <Table.Head>Measured</Table.Head>}
              <Table.Head>Compact sequence</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {preview.sample.rows.map((row) => (
              <Table.Row key={row.variant}>
                <Table.Cell>
                  <span className="mono-value">{row.variant}</span>
                </Table.Cell>
                <Table.Cell>{row.depth}</Table.Cell>
                <Table.Cell>{row.fitness.toFixed(3)}</Table.Cell>
                {preview.comparisonAvailable && (
                  <Table.Cell>
                    {row.measuredFitness === null ? "Missing" : row.measuredFitness.toFixed(3)}
                  </Table.Cell>
                )}
                <Table.Cell>
                  <CompactSequence sequence={row.sequence} parent={preview.source.wtCompact} />
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </LayerCard>
  )
}

function PublishPanel({
  defaultName,
  onPublish,
}: {
  defaultName: string
  onPublish: (name: string) => Promise<void>
}) {
  const [name, setName] = useState(defaultName)
  const [publishing, setPublishing] = useState(false)
  const [publishError, setPublishError] = useState<string | null>(null)

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!name.trim()) return
    setPublishing(true)
    setPublishError(null)
    try {
      await onPublish(name.trim())
    } catch (cause) {
      setPublishError(cause instanceof Error ? cause.message : "Publishing failed.")
    } finally {
      setPublishing(false)
    }
  }

  return (
    <LayerCard className="panel-card publish-card">
      <div>
        <Text variant="heading3" as="h2">
          Save this landscape
        </Text>
        <Text variant="secondary" size="sm">
          Saving creates a permanent ID and adds the landscape to the library.
        </Text>
      </div>
      <form className="publish-form" onSubmit={(event) => void submit(event)}>
        <Input
          size="sm"
          label="Model name"
          maxLength={80}
          value={name}
          onChange={(event) => setName(event.currentTarget.value)}
          error={publishError ?? undefined}
        />
        <Button
          size="sm"
          type="submit"
          variant="primary"
          icon={<Database />}
          loading={publishing}
          disabled={!name.trim() || publishing}
        >
          Save landscape
        </Button>
      </form>
    </LayerCard>
  )
}

function CodePanel({ examples }: { examples: CodeExamples }) {
  return (
    <LayerCard className="panel-card code-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Query the landscape
          </Text>
          <Text variant="secondary" size="sm">
            Copy a local Python build or call the hosted batch endpoint.
          </Text>
        </div>
        <CheckCircle size={24} className="success-icon" weight="fill" />
      </div>
      <Grid variant="2up" gap="sm">
        <div>
          <Text bold>Local Python</Text>
          <pre className="code-preview">
            <code className="language-python">{highlightPython(examples.localPython)}</code>
          </pre>
          <ClipboardText size="sm" text="Copy local Python" textToCopy={examples.localPython} />
        </div>
        <div>
          <Text bold>Hosted API</Text>
          <pre className="code-preview">
            <code className="language-python">{highlightPython(examples.hostedPython)}</code>
          </pre>
          <ClipboardText size="sm" text="Copy hosted Python" textToCopy={examples.hostedPython} />
        </div>
      </Grid>
    </LayerCard>
  )
}

function ColumnMapper({
  inspection,
  mapping,
  onChange,
}: {
  inspection: CsvInspection
  mapping: ColumnMapping
  onChange: (next: ColumnMapping) => void
}) {
  const schema = inspection.schemas.find((entry) => entry.id === mapping.schema)
  const headerItems = Object.fromEntries(inspection.headers.map((header) => [header, header]))

  return (
    <div className="column-mapper">
      <Select
        label="Row layout"
        size="sm"
        value={mapping.schema}
        onValueChange={(value) => {
          const next = inspection.schemas.find((entry) => entry.id === value)
          if (!next) return
          onChange({
            schema: next.id,
            columns: Object.fromEntries(
              next.roles
                .filter((role) => role.guess)
                .map((role) => [role.id, role.guess as string]),
            ),
          })
        }}
        items={Object.fromEntries(
          inspection.schemas.map((entry) => [entry.id, entry.label] as const),
        )}
        className="full-width"
      />
      <div className="column-mapper-roles">
        {schema?.roles.map((role) => (
          <Select
            key={role.id}
            label={role.label}
            size="sm"
            value={mapping.columns[role.id] ?? ""}
            onValueChange={(value) =>
              onChange({
                ...mapping,
                columns: { ...mapping.columns, [role.id]: String(value) },
              })
            }
            items={headerItems}
          />
        ))}
      </div>
      {inspection.sampleRows.length > 0 && (
        <div className="csv-preview">
          <table>
            <thead>
              <tr>
                {inspection.headers.map((header) => (
                  <th key={header}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {inspection.sampleRows.slice(0, 4).map((row, rowIndex) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: sample rows have no identifier
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    // biome-ignore lint/suspicious/noArrayIndexKey: cells are positional
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function CustomUploadPanel({
  onCreated,
}: {
  onCreated: (draft: DraftResponse, name: string) => void
}) {
  const [intake, setIntake] = useState<"files" | "mavedb">("files")
  const [landscapeName, setLandscapeName] = useState("")
  const [dmsCsv, setDmsCsv] = useState("")
  const [dmsName, setDmsName] = useState("")
  const [wtFasta, setWtFasta] = useState("")
  const [fastaName, setFastaName] = useState("")
  const [inspection, setInspection] = useState<CsvInspection | null>(null)
  const [mapping, setMapping] = useState<ColumnMapping | null>(null)
  const [mappingEdited, setMappingEdited] = useState(false)
  const [mapperOpen, setMapperOpen] = useState(false)
  const [wildTypeScore, setWildTypeScore] = useState("")
  const [reference, setReference] = useState("")
  const [record, setRecord] = useState<MaveDbImport | null>(null)
  const [fetching, setFetching] = useState(false)
  const [creating, setCreating] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const complete =
    mapping !== null &&
    (inspection?.schemas
      .find((entry) => entry.id === mapping.schema)
      ?.roles.every((role) => mapping.columns[role.id]) ??
      false)
  const needsBaseline = mapping?.schema === "hgvs" || mapping?.schema === "variants"

  function applyInspection(next: CsvInspection) {
    setInspection(next)
    setMappingEdited(false)
    setMapping(next.detected)
    setMapperOpen(next.detected === null)
  }

  async function acceptCsv(text: string, fileName: string) {
    setDmsCsv(text)
    setDmsName(fileName)
    setInspection(null)
    setMapping(null)
    setUploadError(null)
    try {
      applyInspection(await inspectCsv(text))
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "That CSV could not be read.")
    }
  }

  async function importRecord() {
    setFetching(true)
    setUploadError(null)
    try {
      const next = await importMaveDb(reference)
      setRecord(next)
      setDmsCsv(next.dmsCsv)
      setDmsName(`${next.urn}.csv`)
      setWtFasta(next.wtFasta)
      setFastaName(next.targetName)
      applyInspection(next.inspection)
      setLandscapeName((current) => current || next.title.slice(0, 80))
    } catch (cause) {
      setRecord(null)
      setUploadError(cause instanceof Error ? cause.message : "That MaveDB record failed to load.")
    } finally {
      setFetching(false)
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    setCreating(true)
    setUploadError(null)
    const baseline = Number.parseFloat(wildTypeScore)
    try {
      const next = await createDraft({
        dmsCsv,
        wtFasta,
        columnMap:
          mapping && (mappingEdited || inspection?.detected === null) ? mapping : undefined,
        wildTypeScore: needsBaseline && Number.isFinite(baseline) ? baseline : undefined,
      })
      onCreated(next, landscapeName.trim())
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "The upload failed.")
    } finally {
      setCreating(false)
    }
  }

  return (
    <form className="upload-form source-upload-form" onSubmit={(event) => void submit(event)}>
      {uploadError && <Banner variant="error" title="Upload failed" description={uploadError} />}
      <Tabs
        size="sm"
        variant="segmented"
        value={intake}
        onValueChange={(value) => setIntake(value as "files" | "mavedb")}
        tabs={[
          { value: "files", label: "Files" },
          { value: "mavedb", label: "MaveDB" },
        ]}
        className="source-mode-tabs"
        listClassName="source-mode-tab-list"
      />

      {intake === "files" ? (
        <div className="drop-zone-grid">
          <DropFileField
            label="DMS CSV"
            description="any layout, map the columns below"
            accept=".csv,text/csv"
            fileName={dmsName}
            onFile={(file) => file.text().then((text) => acceptCsv(text, file.name))}
          />
          <DropFileField
            label="WT FASTA"
            description="Parent amino-acid sequence"
            accept=".fasta,.fa,.faa,text/plain"
            fileName={fastaName}
            onFile={async (file) => {
              setWtFasta(await file.text())
              setFastaName(file.name)
            }}
          />
        </div>
      ) : (
        <div className="mavedb-intake">
          <Input
            size="sm"
            label="Score-set URN or URL"
            labelTooltip="Paste a mavedb.org score-set link or a urn:mavedb identifier. The target protein sequence arrives with the scores, so no FASTA is needed."
            placeholder="urn:mavedb:00001222-b-2"
            value={reference}
            onChange={(event) => setReference(event.currentTarget.value)}
          />
          <Button
            size="sm"
            variant="secondary"
            icon={<Download />}
            loading={fetching}
            disabled={!reference.trim() || fetching}
            onClick={() => void importRecord()}
          >
            Fetch score set
          </Button>
          {record && (
            <dl className="compact-facts mavedb-facts">
              <div>
                <dt>Target</dt>
                <dd>{record.targetName}</dd>
              </div>
              <div>
                <dt>Variants</dt>
                <dd>{record.numVariants.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Licence</dt>
                <dd>{record.license || "Not stated"}</dd>
              </div>
            </dl>
          )}
        </div>
      )}

      <Input
        size="sm"
        label="Landscape name"
        labelTooltip="Names this landscape in the sidebar and the library."
        maxLength={80}
        placeholder={record?.title ?? "My DMS landscape"}
        value={landscapeName}
        onChange={(event) => setLandscapeName(event.currentTarget.value)}
      />

      {inspection && mapping && (
        <div className="mapping-summary">
          <Text size="sm">
            {inspection.rowCount.toLocaleString()} rows read as{" "}
            <strong>
              {inspection.schemas.find((entry) => entry.id === mapping.schema)?.label.toLowerCase()}
            </strong>
            .
          </Text>
          <Button size="xs" variant="ghost" onClick={() => setMapperOpen((open) => !open)}>
            {mapperOpen ? "Hide columns" : "Change columns"}
          </Button>
        </div>
      )}
      {inspection && !mapping && (
        <Banner
          size="sm"
          variant="alert"
          title="Columns not recognised"
          description="Pick the row layout and map each column by hand."
        />
      )}
      {inspection && mapperOpen && (
        <ColumnMapper
          inspection={inspection}
          mapping={mapping ?? { schema: inspection.schemas[0].id, columns: {} }}
          onChange={(next) => {
            setMappingEdited(true)
            setMapping(next)
          }}
        />
      )}
      {needsBaseline && (
        <Input
          size="sm"
          type="number"
          step="any"
          label="Wild-type score"
          labelTooltip="The raw score a wild-type variant carries in this file. Every fitness is shifted so wild type sits at 0. Leave blank to use the file's own wild-type or synonymous rows, or 0 when it has none."
          placeholder="From the file"
          value={wildTypeScore}
          onChange={(event) => setWildTypeScore(event.currentTarget.value)}
        />
      )}

      <div className="source-upload-action">
        <Text variant="secondary" size="xs">
          Uploaded data remains a temporary draft until you save the landscape.
        </Text>
        <Button
          type="submit"
          size="sm"
          variant="primary"
          loading={creating}
          disabled={!dmsCsv || !wtFasta || !complete || creating}
        >
          Use this data
        </Button>
      </div>
    </form>
  )
}

function DropFileField({
  label,
  description,
  required = false,
  accept,
  fileName,
  onFile,
}: {
  label: string
  description: string
  required?: boolean
  accept: string
  fileName: string
  onFile: (file: File) => void | Promise<void>
}) {
  const [dragging, setDragging] = useState(false)

  function acceptDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) void onFile(file)
  }

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]
    if (file) void onFile(file)
  }

  return (
    <label
      className={dragging ? "drop-file-field dragging" : "drop-file-field"}
      onDragEnter={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={acceptDrop}
    >
      <input type="file" accept={accept} onChange={selectFile} required={required} />
      <UploadSimple size={22} weight={fileName ? "fill" : "regular"} />
      <span className="drop-file-copy">
        <strong>{label}</strong>
        <small>{fileName || `Drop file here or browse · ${description}`}</small>
      </span>
      {fileName && <CheckCircle className="success-icon" size={18} weight="fill" />}
    </label>
  )
}

function ModelCatalogue({
  models,
  loading,
  onRefresh,
  onExplore,
}: {
  models: ModelSummary[]
  loading: boolean
  onRefresh: () => Promise<void>
  onExplore: (modelId: string) => void
}) {
  return (
    <div className="studio-stack">
      <section className="page-heading">
        <div>
          <Text variant="heading2" as="h1">
            Landscape library
          </Text>
          <Text variant="secondary">Frozen references and saved custom benchmark landscapes.</Text>
        </div>
        <Button
          size="sm"
          variant="secondary"
          icon={<ArrowClockwise />}
          loading={loading}
          onClick={() => void onRefresh()}
        >
          Refresh
        </Button>
      </section>
      <LayerCard className="panel-card catalogue-card">
        <div className="table-scroll">
          <Table>
            <Table.Header>
              <Table.Row>
                <Table.Head>Model</Table.Head>
                <Table.Head>Type</Table.Head>
                <Table.Head>Sites</Table.Head>
                <Table.Head>Parent</Table.Head>
                <Table.Head>Published</Table.Head>
                <Table.Head />
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {models.map((model) => (
                <Table.Row key={model.id}>
                  <Table.Cell>
                    <div className="model-cell">
                      <Text bold>{model.name}</Text>
                      <Text variant="secondary" size="xs">
                        {model.id}
                      </Text>
                    </div>
                  </Table.Cell>
                  <Table.Cell>
                    <Badge variant={model.kind === "reference" ? "blue" : "purple"}>
                      {model.kind}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>{model.positions.length}</Table.Cell>
                  <Table.Cell>
                    <span className="mono-value">{model.wtCompact}</span>
                  </Table.Cell>
                  <Table.Cell>
                    {model.createdAt ? new Date(model.createdAt).toLocaleDateString() : "Frozen"}
                  </Table.Cell>
                  <Table.Cell>
                    <Button size="sm" variant="secondary" onClick={() => onExplore(model.id)}>
                      Explore
                    </Button>
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table>
        </div>
      </LayerCard>
    </div>
  )
}

export function Leaderboards({ models }: { models: ModelSummary[] }) {
  const [boards, setBoards] = useState<Leaderboard[]>([])
  const [selectedBoardId, setSelectedBoardId] = useState("")
  const [loading, setLoading] = useState(true)
  const [boardError, setBoardError] = useState<string | null>(null)

  const refreshBoards = useCallback(async (preferredId?: string) => {
    setLoading(true)
    setBoardError(null)
    try {
      const next = await listLeaderboards()
      setBoards(next)
      setSelectedBoardId((current) =>
        preferredId && next.some((board) => board.id === preferredId)
          ? preferredId
          : next.some((board) => board.id === current)
            ? current
            : (next[0]?.id ?? ""),
      )
    } catch (cause) {
      setBoardError(cause instanceof Error ? cause.message : "Could not load leaderboards.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshBoards()
  }, [refreshBoards])

  const selectedBoard = boards.find((board) => board.id === selectedBoardId) ?? null
  const landscapeName =
    models.find((model) => model.id === selectedBoard?.landscapeId)?.name ??
    selectedBoard?.landscapeId

  async function makeBoard(name: string, landscapeId: string, queryBudget: number) {
    const board = await createLeaderboard(name, landscapeId, queryBudget)
    await refreshBoards(board.id)
  }

  async function addResult(modelName: string, successes: number, trials: number) {
    if (!selectedBoard) return
    await addLeaderboardEntry(selectedBoard.id, modelName, successes, trials)
    await refreshBoards(selectedBoard.id)
  }

  return (
    <div className="studio-stack">
      <section className="page-heading">
        <div>
          <Text variant="heading2" as="h1">
            Leaderboards
          </Text>
          <Text variant="secondary">
            Compare model success rates under one landscape and query budget.
          </Text>
        </div>
        <CreateLeaderboardDialog models={models} onCreate={makeBoard} />
      </section>
      {boardError && (
        <Banner variant="error" title="Leaderboard request failed" description={boardError} />
      )}
      {loading && boards.length === 0 ? (
        <LayerCard className="leaderboard-empty panel-card">
          <Empty size="base" icon={<Trophy size={40} />} title="Loading leaderboards" />
        </LayerCard>
      ) : boards.length === 0 ? (
        <LayerCard className="leaderboard-empty panel-card">
          <Empty
            size="base"
            icon={<Trophy size={40} />}
            title="Create your first leaderboard"
            description="Fix a landscape and query budget, then add results from your models."
          />
        </LayerCard>
      ) : (
        <div className="leaderboard-layout">
          <LayerCard className="panel-card board-list-card">
            <Text variant="heading3" as="h2">
              Boards
            </Text>
            <div className="board-list">
              {boards.map((board) => (
                <button
                  key={board.id}
                  type="button"
                  className={board.id === selectedBoardId ? "board-item active" : "board-item"}
                  onClick={() => setSelectedBoardId(board.id)}
                >
                  <span>{board.name}</span>
                  <small>{board.entries.length} models</small>
                </button>
              ))}
            </div>
          </LayerCard>
          {selectedBoard && (
            <div className="results-stack">
              <LayerCard className="panel-card board-summary-card">
                <div className="panel-heading">
                  <div>
                    <Text variant="heading3" as="h2">
                      {selectedBoard.name}
                    </Text>
                    <Text variant="secondary" size="sm">
                      {landscapeName} · {selectedBoard.queryBudget.toLocaleString()} queries per run
                    </Text>
                  </div>
                  <AddResultDialog onAdd={addResult} />
                </div>
                {selectedBoard.entries.length > 0 ? (
                  document.documentElement.clientWidth > 0 && (
                    <Chart
                      echarts={echarts}
                      options={leaderboardChartOptions(selectedBoard.entries)}
                      height={Math.max(190, selectedBoard.entries.length * 38 + 70)}
                    />
                  )
                ) : (
                  <div className="board-empty-state">
                    <ChartBar size={32} />
                    <Text variant="secondary" size="sm">
                      Add a model result to start the comparison.
                    </Text>
                  </div>
                )}
              </LayerCard>
              {selectedBoard.entries.length > 0 && (
                <LeaderboardTable entries={selectedBoard.entries} />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function leaderboardChartOptions(entries: LeaderboardEntry[]): KumoChartOption {
  return {
    animation: false,
    grid: { left: 126, right: 44, top: 8, bottom: 24 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (value: unknown) =>
        typeof value === "number" ? `${value.toFixed(1)}%` : String(value),
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 100,
      axisLabel: { formatter: "{value}%" },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: entries.map((entry) => entry.modelName),
      axisLabel: { width: 110, overflow: "truncate" },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        type: "bar",
        name: "Success rate",
        data: entries.map((entry) => Number((entry.successRate * 100).toFixed(2))),
        barMaxWidth: 22,
        itemStyle: { color: "#27ae60", borderRadius: [0, 4, 4, 0] },
        label: { show: true, position: "right", formatter: "{c}%" },
      },
    ],
  }
}

function LeaderboardTable({ entries }: { entries: LeaderboardEntry[] }) {
  return (
    <LayerCard className="panel-card catalogue-card">
      <div className="table-scroll">
        <Table>
          <Table.Header>
            <Table.Row>
              <Table.Head>Rank</Table.Head>
              <Table.Head>Model</Table.Head>
              <Table.Head>Successes</Table.Head>
              <Table.Head>Success rate</Table.Head>
              <Table.Head>95% interval</Table.Head>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {entries.map((entry, index) => (
              <Table.Row key={entry.id}>
                <Table.Cell>{index + 1}</Table.Cell>
                <Table.Cell>
                  <Text bold>{entry.modelName}</Text>
                </Table.Cell>
                <Table.Cell>
                  {entry.successes}/{entry.trials}
                </Table.Cell>
                <Table.Cell>{(entry.successRate * 100).toFixed(1)}%</Table.Cell>
                <Table.Cell>
                  {(entry.confidenceLow * 100).toFixed(1)}% to{" "}
                  {(entry.confidenceHigh * 100).toFixed(1)}%
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </LayerCard>
  )
}

function CreateLeaderboardDialog({
  models,
  onCreate,
}: {
  models: ModelSummary[]
  onCreate: (name: string, landscapeId: string, queryBudget: number) => Promise<void>
}) {
  const [name, setName] = useState("")
  const [landscapeId, setLandscapeId] = useState(models[0]?.id ?? "")
  const [queryBudget, setQueryBudget] = useState(1_000)
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const items = Object.fromEntries(models.map((model) => [model.id, model.name]))

  useEffect(() => {
    if (!landscapeId && models[0]) setLandscapeId(models[0].id)
  }, [landscapeId, models])

  async function submit(event: FormEvent) {
    event.preventDefault()
    setSaving(true)
    setFormError(null)
    try {
      await onCreate(name.trim(), landscapeId, queryBudget)
      setName("")
      setOpen(false)
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not create leaderboard.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger
        render={(triggerProps) => (
          <Button {...triggerProps} size="sm" variant="primary" icon={<Plus />}>
            Create leaderboard
          </Button>
        )}
      />
      <Dialog size="base" className="upload-dialog">
        <Dialog.Title>
          <Text variant="heading2" as="span">
            Create leaderboard
          </Text>
        </Dialog.Title>
        <Dialog.Description>
          Every result on this board uses the same landscape and query budget.
        </Dialog.Description>
        <form className="upload-form dialog-form" onSubmit={(event) => void submit(event)}>
          <Input
            size="sm"
            label="Board name"
            value={name}
            maxLength={80}
            onChange={(event) => setName(event.currentTarget.value)}
          />
          <Select
            size="sm"
            label="Landscape"
            items={items}
            value={landscapeId}
            onValueChange={(value) => setLandscapeId(String(value))}
            className="full-width"
          />
          <Input
            size="sm"
            label="Queries per run"
            type="number"
            min={1}
            max={10_000_000}
            value={queryBudget}
            onChange={(event) => setQueryBudget(Number(event.currentTarget.value))}
            error={formError ?? undefined}
          />
          <div className="dialog-actions">
            <Dialog.Close
              render={(closeProps) => (
                <Button {...closeProps} size="sm" type="button" variant="secondary">
                  Cancel
                </Button>
              )}
            />
            <Button
              size="sm"
              type="submit"
              variant="primary"
              loading={saving}
              disabled={!name.trim() || !landscapeId || saving}
            >
              Create board
            </Button>
          </div>
        </form>
      </Dialog>
    </Dialog.Root>
  )
}

function AddResultDialog({
  onAdd,
}: {
  onAdd: (modelName: string, successes: number, trials: number) => Promise<void>
}) {
  const [modelName, setModelName] = useState("")
  const [successes, setSuccesses] = useState(0)
  const [trials, setTrials] = useState(10)
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setSaving(true)
    setFormError(null)
    try {
      await onAdd(modelName.trim(), successes, trials)
      setModelName("")
      setOpen(false)
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not add result.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger
        render={(triggerProps) => (
          <Button {...triggerProps} size="sm" variant="primary" icon={<Plus />}>
            Add model result
          </Button>
        )}
      />
      <Dialog size="base" className="upload-dialog">
        <Dialog.Title>
          <Text variant="heading2" as="span">
            Add model result
          </Text>
        </Dialog.Title>
        <Dialog.Description>Record repeated benchmark runs for one model.</Dialog.Description>
        <form className="upload-form dialog-form" onSubmit={(event) => void submit(event)}>
          <Input
            size="sm"
            label="Model name"
            value={modelName}
            maxLength={120}
            onChange={(event) => setModelName(event.currentTarget.value)}
          />
          <div className="field-grid">
            <Input
              size="sm"
              label="Successful runs"
              type="number"
              min={0}
              max={trials}
              value={successes}
              onChange={(event) => setSuccesses(Number(event.currentTarget.value))}
            />
            <Input
              size="sm"
              label="Total runs"
              type="number"
              min={1}
              max={1_000_000}
              value={trials}
              onChange={(event) => setTrials(Number(event.currentTarget.value))}
              error={formError ?? undefined}
            />
          </div>
          <div className="dialog-actions">
            <Dialog.Close
              render={(closeProps) => (
                <Button {...closeProps} size="sm" type="button" variant="secondary">
                  Cancel
                </Button>
              )}
            />
            <Button
              size="sm"
              type="submit"
              variant="primary"
              loading={saving}
              disabled={!modelName.trim() || successes > trials || saving}
            >
              Add result
            </Button>
          </div>
        </form>
      </Dialog>
    </Dialog.Root>
  )
}

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
} from "@cloudflare/kumo"
import {
  ArrowClockwise,
  ArrowLeft,
  ChartBar,
  CheckCircle,
  Database,
  Dna,
  Flask,
  Play,
  Plus,
  SquaresFour,
  Trophy,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react"
import { BarChart, GraphChart, LineChart } from "echarts/charts"
import {
  AriaComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components"
import * as echarts from "echarts/core"
import { CanvasRenderer } from "echarts/renderers"
import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
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
  getModel,
  listLeaderboards,
  listModels,
  publishModel,
  queryPosterior,
  runPosteriorPreview,
  runPreview,
} from "./api"
import {
  benchmarkBudgetOptions,
  benchmarkValidityOptions,
  fitnessProfileOptions,
  landscapeGraphOptions,
  posteriorProfileOptions,
  walshOptions,
} from "./charts"
import type {
  CodeExamples,
  DraftResponse,
  EvaluationResult,
  Leaderboard,
  LeaderboardEntry,
  ModelParameters,
  ModelSummary,
  PosteriorPreviewResponse,
  PosteriorQueryResult,
  PreviewResponse,
  PreviewSource,
} from "./types"

echarts.use([
  LineChart,
  BarChart,
  GraphChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
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

export type Page = "studio" | "explore" | "models" | "leaderboards"

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

function sampleCapacity(positionCount: number, maxDepth: number): number {
  let combinations = 1
  let total = 0
  for (let depth = 1; depth <= Math.min(positionCount, maxDepth); depth++) {
    combinations = (combinations * (positionCount - depth + 1)) / depth
    total += combinations * 19 ** depth
  }
  return Math.min(100_000, total)
}

function modelSampleLimit(model: ModelSummary | null, maxDepth: number): number {
  if (!model) return 1
  const theoretical = sampleCapacity(model.positions.length, maxDepth)
  if (model.kind !== "posterior") return theoretical
  return Math.min(5_000, model.sourceSummary?.previewCapacity ?? theoretical)
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
  const [selectedId, setSelectedId] = useState("gb1_headline")
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
  const [error, setError] = useState<string | null>(null)
  const [publishedExamples, setPublishedExamples] = useState<CodeExamples | null>(null)

  const availableModels = useMemo(
    () => (draftModel ? [draftModel, ...models] : models),
    [draftModel, models],
  )
  const selectedModel = availableModels.find((model) => model.id === selectedId) ?? null
  const selectedIsDraft = draft !== null && selectedId === draft.id
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
      setSelectedId((current) =>
        !items.some((model) => model.id === current) && current !== draft?.id
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
  }, [selectedId, selectedIsDraft])

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
      Math.min(model.kind === "posterior" ? 500 : 10_000, modelSampleLimit(model, nextMaxDepth)),
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
      Math.min(model.kind === "posterior" ? 500 : 10_000, modelSampleLimit(model, nextMaxDepth)),
    )
    setPreview(null)
    setPosteriorPreview(null)
    setError(null)
  }

  function changeMaxDepth(value: number) {
    const positionCount = selectedModel?.positions.length ?? 1
    const bounded = Math.max(1, Math.min(positionCount, Math.floor(value || 1)))
    setMaxDepth(bounded)
    setSampleSize((current) => Math.min(current, modelSampleLimit(selectedModel, bounded)))
    setDepthWeights((current) => {
      const next: Record<number, number> = {}
      for (let depth = 1; depth <= bounded; depth++) next[depth] = current[depth] ?? 1
      return next
    })
  }

  async function applyPreview(): Promise<boolean> {
    if (!selectedModel) return false
    setRunning(true)
    setError(null)
    try {
      const source = sourceFromModel(selectedModel, draft?.id)
      if (selectedIsPosterior) {
        const result = await runPosteriorPreview(
          source,
          priorId,
          oracleSeed,
          ensembleSize,
          sampleSize,
          depthWeights,
        )
        setPosteriorPreview(result)
        setPreview(null)
      } else {
        const result = await runPreview(source, parameters, sampleSize, depthWeights)
        setPreview(result)
        setPosteriorPreview(null)
      }
      return true
    } catch (cause) {
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

  function acceptDraft(created: DraftResponse) {
    const summary = created.summary
    const model: ModelSummary = {
      id: created.id,
      name: "Custom draft",
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
    setSampleSize(Math.min(500, summary.previewCapacity))
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
            </Text>
          </div>
          <Badge variant="beta">Research preview</Badge>
        </header>

        <main className="page-frame">
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
              createLandscape={createLandscape}
              acceptDraft={acceptDraft}
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
              sampleLimit={modelSampleLimit(selectedModel, maxDepth)}
              setSampleSize={setSampleSize}
              maxDepth={maxDepth}
              changeMaxDepth={changeMaxDepth}
              depthWeights={depthWeights}
              setDepthWeights={setDepthWeights}
              running={running}
              applyPreview={applyPreview}
              draft={draft}
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
  createLandscape: () => Promise<void>
  acceptDraft: (draft: DraftResponse) => void
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
    createLandscape,
    acceptDraft,
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
              <Text variant="secondary" size="sm">
                Choose existing data or upload your own.
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
              { value: "preset", label: "Preset data" },
              { value: "upload", label: "Upload DMS" },
            ]}
            className="source-mode-tabs"
            listClassName="source-mode-tab-list"
          />
          {sourceMode === "preset" ? (
            <div className="source-mode-panel">
              <Select
                label="Landscape source"
                size="sm"
                value={selectedModel?.id}
                onValueChange={(value) => value && selectModel(String(value))}
                items={modelItems}
                loading={loadingModels}
                className="full-width"
              />
              {selectedModel && (
                <div className="source-summary">
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
                  </dl>
                </div>
              )}
            </div>
          ) : (
            <CustomUploadPanel
              onCreated={(created) => {
                acceptDraft(created)
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
            hasStructure={hasStructure}
            canFitNonlinearity={canFitNonlinearity}
          />
        )}
      </Grid>

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
  setDepthWeights: (
    update: Record<number, number> | ((current: Record<number, number>) => Record<number, number>),
  ) => void
  running: boolean
  applyPreview: () => Promise<boolean>
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
          publish={publish}
          examples={publishedExamples}
        />
      ) : preview ? (
        <PreviewResults
          preview={preview}
          draft={draft}
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
          <Text variant="secondary" size="sm">
            Enter a mutation set such as {mutationExample(selectedModel)}. Use WT for the parent.
          </Text>
        </div>
      </div>
      <form className="query-form" onSubmit={(event) => void submit(event)}>
        <Input
          size="sm"
          label="Mutation set"
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
      {latestPosterior && (
        <div className="posterior-query-results" aria-live="polite">
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
  applyPreview,
  hasPreview,
}: SamplingProps & { hasPreview: boolean }) {
  return (
    <LayerCard className="panel-card sample-controls">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Sample the landscape
          </Text>
          <Text variant="secondary" size="sm">
            Set the unique sample size and mutation-depth mix.
          </Text>
        </div>
      </div>
      <div className="sample-control-grid">
        <Input
          size="sm"
          label="Variants"
          type="number"
          min={1}
          max={sampleLimit}
          value={sampleSize}
          onChange={(event) => setSampleSize(Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Maximum depth"
          type="number"
          min={1}
          max={selectedModel?.positions.length ?? 1}
          value={maxDepth}
          onChange={(event) => changeMaxDepth(Number(event.currentTarget.value))}
        />
        {Object.keys(depthWeights).map((rawDepth) => {
          const depth = Number(rawDepth)
          return (
            <Input
              size="sm"
              key={depth}
              label={`Depth ${depth}`}
              type="number"
              min={0}
              step={0.1}
              value={depthWeights[depth]}
              onChange={(event) =>
                setDepthWeights((current) => ({
                  ...current,
                  [depth]: Number(event.currentTarget.value),
                }))
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
      {running && (
        <Text variant="secondary" size="xs">
          Complete references can take about 20 seconds on a cold run.
        </Text>
      )}
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
          <Text variant="secondary" size="sm">
            Set the readout calibration, stable oracle seed, and ensemble size.
          </Text>
        </div>
        {locked && <Badge variant="neutral">Fixed oracle</Badge>}
      </div>
      <div className="field-grid">
        <Select
          size="sm"
          label="Readout calibration"
          value={priorId}
          items={{ "combinatorial-v2": "Combinatorial v2" }}
          disabled={locked}
          onValueChange={(value) => value && setPriorId(String(value))}
        />
        <Input
          size="sm"
          label="Oracle seed"
          type="number"
          step={1}
          value={oracleSeed}
          disabled={locked}
          onChange={(event) => setOracleSeed(Number(event.currentTarget.value))}
        />
        <Input
          size="sm"
          label="Oracle draws"
          labelTooltip="More draws give smoother ensemble ranges and take longer."
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
        title="Transferred empirical prior"
        description="Version one pools pairwise interaction statistics from 30 landscapes across 14 proteins."
      />
    </LayerCard>
  )
}

interface ParameterPanelProps {
  value: ModelParameters
  onChange: (value: ModelParameters) => void
  locked: boolean
  hasStructure: boolean
  canFitNonlinearity: boolean
}

function ParameterPanel({
  value,
  onChange,
  locked,
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
          <Text variant="secondary" size="sm">
            Set the fitness readout, pairwise effects, and repeatable seed.
          </Text>
        </div>
        {locked && <Badge variant="neutral">Fixed</Badge>}
      </div>
      <div className="field-grid">
        <Input
          size="sm"
          label="Ruggedness"
          labelTooltip="Gaussian genotype-stable noise before the fitness readout."
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
          labelTooltip="Positive range available to combined beneficial mutations."
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
          type="number"
          step={1}
          value={value.seed}
          disabled={locked}
          onChange={(event) => set("seed", Number(event.currentTarget.value))}
        />
      </div>
      {!hasStructure && (
        <Banner
          size="sm"
          variant="secondary"
          title="No structural contacts"
          description="Pairwise sigma and alpha stay inactive for this model."
        />
      )}
      {!canFitNonlinearity && (
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

function PreviewResults({
  preview,
  draft,
  publish,
  examples,
}: {
  preview: PreviewResponse
  draft: DraftResponse | null
  publish: (name: string) => Promise<void>
  examples: CodeExamples | null
}) {
  return (
    <div className="results-stack">
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

      <LayerCard className="panel-card chart-card">
        <div className="panel-heading">
          <div>
            <Text variant="heading3" as="h2">
              Interactive landscape map
            </Text>
            <Text variant="secondary" size="sm">
              Paths run from the parent to sampled variants. Move right for mutation depth and up
              for higher fitness. Drag to pan and scroll to zoom.
            </Text>
          </div>
          <Badge variant="neutral">{preview.sample.graph.nodes.length} variants</Badge>
        </div>
        <Chart
          echarts={echarts}
          options={landscapeGraphOptions(preview.sample.graph)}
          height={430}
        />
      </LayerCard>

      {preview.notices.map((notice) => (
        <Banner key={notice} size="sm" variant="secondary" description={notice} />
      ))}

      {preview.walsh && (
        <Grid variant="2up" gap="sm">
          <LayerCard className="panel-card chart-card">
            <Text variant="heading3" as="h2">
              Interaction order
            </Text>
            <Text variant="secondary" size="sm">
              Fraction of variance assigned to each Walsh order.
            </Text>
            <Chart echarts={echarts} options={walshOptions(preview.walsh)} height={250} />
          </LayerCard>
          <MetricTable rows={preview.metrics} />
        </Grid>
      )}

      <SampleTable preview={preview} />

      {draft && <PublishPanel onPublish={publish} />}
      {examples && <CodePanel examples={examples} />}
    </div>
  )
}

function PosteriorPreviewResults({
  preview,
  draft,
  publish,
  examples,
}: {
  preview: PosteriorPreviewResponse
  draft: DraftResponse | null
  publish: (name: string) => Promise<void>
  examples: CodeExamples | null
}) {
  return (
    <div className="results-stack">
      <Banner
        size="sm"
        variant="secondary"
        title={preview.prior.name}
        description={`The readout calibration uses ${preview.prior.nLandscapes} landscapes across ${preview.prior.nProteins} proteins. Each draw adds stable random pair interactions at coupling ${preview.settings.coupling}.`}
      />
      <LayerCard className="panel-card chart-card">
        <div className="panel-heading">
          <div>
            <Text variant="heading3" as="h2">
              Does the benchmark rank search methods correctly?
            </Text>
            <Text variant="secondary" size="sm">
              {preview.benchmarkValidity.nMethods} methods, budget{" "}
              {preview.benchmarkValidity.budget}, across {preview.benchmarkValidity.nLandscapes}{" "}
              complete development landscapes.
            </Text>
          </div>
          <Badge variant="green">Task transfer</Badge>
        </div>
        <Chart
          echarts={echarts}
          options={benchmarkValidityOptions(preview.benchmarkValidity)}
          height={360}
        />
        <Banner
          size="sm"
          variant="secondary"
          title={`${preview.benchmarkValidity.external.nLandscapes}-landscape external protein panel`}
          description={`Rank transfer ${preview.benchmarkValidity.external.kendallTau.toFixed(3)}, pairwise winner accuracy ${(100 * preview.benchmarkValidity.external.pairwiseAccuracy).toFixed(1)}%, reliability pass rate ${(100 * preview.benchmarkValidity.external.reliabilityPassRate).toFixed(1)}%.`}
        />
      </LayerCard>
      {preview.benchmarkValidity.budgetCurve && (
        <LayerCard className="panel-card chart-card">
          <div className="panel-heading">
            <div>
              <Text variant="heading3" as="h2">
                Do reliable method decisions persist as the query budget grows?
              </Text>
              <Text variant="secondary" size="sm">
                Solid lines show the development panel. Dashed lines show the external protein
                panel. Accuracy pools method pairs with a reliable measured winner.
              </Text>
            </div>
            <Badge variant="blue">Budget sensitivity</Badge>
          </div>
          <Chart
            echarts={echarts}
            options={benchmarkBudgetOptions(preview.benchmarkValidity)}
            height={380}
          />
        </LayerCard>
      )}
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
        <Chart echarts={echarts} options={posteriorProfileOptions(preview.profile)} height={340} />
      </LayerCard>

      <LayerCard className="panel-card chart-card">
        <div className="panel-heading">
          <div>
            <Text variant="heading3" as="h2">
              Synthetic landscape map
            </Text>
            <Text variant="secondary" size="sm">
              Colour shows fitness. Larger purple borders show greater variation across draws.
            </Text>
          </div>
          <Badge variant="neutral">{preview.sample.graph.nodes.length} variants</Badge>
        </div>
        <Chart
          echarts={echarts}
          options={landscapeGraphOptions(preview.sample.graph)}
          height={430}
        />
      </LayerCard>

      {preview.notices.map((notice) => (
        <Banner key={notice} size="sm" variant="secondary" description={notice} />
      ))}

      <PosteriorSampleTable preview={preview} />
      {draft && <PublishPanel onPublish={publish} />}
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
                  <span className="mono-value">{row.sequence}</span>
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
                  <span className="mono-value">{row.sequence}</span>
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table>
      </div>
    </LayerCard>
  )
}

function PublishPanel({ onPublish }: { onPublish: (name: string) => Promise<void> }) {
  const [name, setName] = useState("")
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
          <pre className="code-preview">{examples.localPython}</pre>
          <ClipboardText size="sm" text="Copy local Python" textToCopy={examples.localPython} />
        </div>
        <div>
          <Text bold>Hosted API</Text>
          <pre className="code-preview">{examples.hostedPython}</pre>
          <ClipboardText size="sm" text="Copy hosted Python" textToCopy={examples.hostedPython} />
        </div>
      </Grid>
    </LayerCard>
  )
}

function CustomUploadPanel({ onCreated }: { onCreated: (draft: DraftResponse) => void }) {
  type StructureMode = "none" | "upload" | "esmfold"
  const [dmsCsv, setDmsCsv] = useState("")
  const [dmsName, setDmsName] = useState("")
  const [wtFasta, setWtFasta] = useState("")
  const [fastaName, setFastaName] = useState("")
  const [structureMode, setStructureMode] = useState<StructureMode>("none")
  const [structureText, setStructureText] = useState("")
  const [structureName, setStructureName] = useState("")
  const [structureFormat, setStructureFormat] = useState<"pdb" | "cif" | "mmcif" | undefined>()
  const [creating, setCreating] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  async function readFile(
    file: File,
    setter: (text: string) => void,
    setName: (name: string) => void,
  ) {
    setter(await file.text())
    setName(file.name)
  }

  async function readStructure(file: File) {
    const extension = file.name.split(".").pop()?.toLowerCase()
    setStructureFormat(extension === "cif" || extension === "mmcif" ? extension : "pdb")
    await readFile(file, setStructureText, setStructureName)
  }

  function chooseStructureMode(value: StructureMode) {
    setStructureMode(value)
    if (value !== "upload") {
      setStructureText("")
      setStructureName("")
      setStructureFormat(undefined)
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    setCreating(true)
    setUploadError(null)
    try {
      const next = await createDraft({
        dmsCsv,
        wtFasta,
        structureText: structureMode === "upload" ? structureText || undefined : undefined,
        structureFormat: structureMode === "upload" ? structureFormat : undefined,
        structureInference: structureMode === "esmfold" ? "esmfold" : "none",
      })
      onCreated(next)
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "The upload failed.")
    } finally {
      setCreating(false)
    }
  }

  return (
    <form className="upload-form source-upload-form" onSubmit={(event) => void submit(event)}>
      {uploadError && <Banner variant="error" title="Upload failed" description={uploadError} />}
      <div className="drop-zone-grid">
        <DropFileField
          label="DMS CSV"
          description="single-site columns or variant and fitness columns"
          accept=".csv,text/csv"
          fileName={dmsName}
          required
          onFile={(file) => readFile(file, setDmsCsv, setDmsName)}
        />
        <DropFileField
          label="WT FASTA"
          description="Parent amino-acid sequence"
          accept=".fasta,.fa,.faa,text/plain"
          fileName={fastaName}
          required
          onFile={(file) => readFile(file, setWtFasta, setFastaName)}
        />
      </div>
      <Banner
        size="sm"
        variant="secondary"
        title="Two supported DMS formats"
        description="Use position, wildtype, mutant, fitness for singles. Use variant, fitness for WT, singles, and optional combinations."
      />
      <Select
        label="Structure source"
        size="sm"
        value={structureMode}
        onValueChange={(value) => chooseStructureMode(String(value) as StructureMode)}
        items={{
          none: "No structure",
          upload: "Upload PDB or mmCIF",
          esmfold: "Infer with ESMFold",
        }}
        className="full-width"
      />
      {structureMode === "upload" && (
        <DropFileField
          label="Structure"
          description="PDB, CIF, or mmCIF"
          accept=".pdb,.cif,.mmcif,chemical/x-pdb"
          fileName={structureName}
          required
          onFile={readStructure}
        />
      )}
      {structureMode === "esmfold" && (
        <Banner
          variant="secondary"
          title="ESMFold inference"
          description="The server predicts and caches the parent structure. The limit is 400 residues."
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
          disabled={
            !dmsCsv || !wtFasta || (structureMode === "upload" && !structureText) || creating
          }
        >
          Use uploaded data
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
        itemStyle: { color: "#f6821f", borderRadius: [0, 4, 4, 0] },
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

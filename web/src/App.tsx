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
  LayerCard,
  Select,
  Sidebar,
  Table,
  Tabs,
  Text,
  type KumoChartOption,
} from "@cloudflare/kumo";
import {
  ArrowLeft,
  ArrowClockwise,
  CheckCircle,
  ChartBar,
  Database,
  Dna,
  Flask,
  Play,
  Plus,
  SquaresFour,
  Trophy,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react";
import { BarChart, GraphChart, LineChart } from "echarts/charts";
import {
  AriaComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  addLeaderboardEntry,
  createDraft,
  createLeaderboard,
  evaluateLandscape,
  getModel,
  listLeaderboards,
  listModels,
  publishModel,
  runPreview,
} from "./api";
import { fitnessProfileOptions, landscapeGraphOptions, walshOptions } from "./charts";
import type {
  CodeExamples,
  DraftResponse,
  EvaluationResult,
  Leaderboard,
  LeaderboardEntry,
  ModelParameters,
  ModelSummary,
  PreviewResponse,
  PreviewSource,
} from "./types";

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
]);

const DEFAULT_PARAMETERS: ModelParameters = {
  noiseSigma: 0.3,
  alpha: 1.5,
  pairwiseSigma: 0,
  combinatorialUpside: 3,
  seed: 42,
};

export type Page = "studio" | "explore" | "models" | "leaderboards";

interface AppProps {
  page?: Page;
  landscapeId?: string;
  onPageChange?: (page: Page) => void;
  onExploreLandscape?: (modelId: string) => void;
}

function sourceFromModel(model: ModelSummary, draftId?: string): PreviewSource {
  if (draftId && model.id === draftId) return { kind: "draft", id: model.id };
  return { kind: model.kind === "reference" ? "reference" : "model", id: model.id };
}

function initialWeights(maxDepth: number): Record<number, number> {
  return Object.fromEntries(Array.from({ length: maxDepth }, (_, index) => [index + 1, 1]));
}

function sampleCapacity(positionCount: number, maxDepth: number): number {
  let combinations = 1;
  let total = 0;
  for (let depth = 1; depth <= Math.min(positionCount, maxDepth); depth++) {
    combinations = (combinations * (positionCount - depth + 1)) / depth;
    total += combinations * 19 ** depth;
  }
  return Math.min(100_000, total);
}

function formatNumber(value: number | null, metricName?: string): string {
  if (value === null) return "N/A";
  if (metricName === "Local optima") return Math.round(value).toLocaleString();
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
  return value.toFixed(3);
}

export default function App({
  page = "studio",
  landscapeId,
  onPageChange,
  onExploreLandscape,
}: AppProps) {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [draft, setDraft] = useState<DraftResponse | null>(null);
  const [draftModel, setDraftModel] = useState<ModelSummary | null>(null);
  const [selectedId, setSelectedId] = useState("gb1_headline");
  const [parameters, setParameters] = useState<ModelParameters>(DEFAULT_PARAMETERS);
  const [sampleSize, setSampleSize] = useState(10_000);
  const [maxDepth, setMaxDepth] = useState(4);
  const [depthWeights, setDepthWeights] = useState<Record<number, number>>(
    initialWeights(4),
  );
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [publishedExamples, setPublishedExamples] = useState<CodeExamples | null>(null);

  const availableModels = useMemo(
    () => (draftModel ? [draftModel, ...models] : models),
    [draftModel, models],
  );
  const selectedModel = availableModels.find((model) => model.id === selectedId) ?? null;
  const selectedIsDraft = draft !== null && selectedId === draft.id;
  const parametersLocked = selectedModel?.kind === "custom" && !selectedIsDraft;
  const hasStructure =
    selectedModel?.kind === "reference" || selectedModel?.sourceSummary?.hasStructure === true;
  const canFitNonlinearity =
    selectedModel?.kind === "reference" ||
    selectedModel?.sourceSummary?.canFitNonlinearity === true;

  async function refreshModels() {
    setLoadingModels(true);
    try {
      const items = await listModels();
      setModels(items);
      if (!items.some((model) => model.id === selectedId) && !selectedIsDraft) {
        setSelectedId(items[0]?.id ?? "");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load the model catalogue.");
    } finally {
      setLoadingModels(false);
    }
  }

  useEffect(() => {
    void refreshModels();
  }, []);

  useEffect(() => {
    if (!selectedId || selectedIsDraft) {
      setPublishedExamples(null);
      return;
    }
    let active = true;
    void getModel(selectedId)
      .then((model) => {
        if (active) setPublishedExamples(model.examples ?? null);
      })
      .catch(() => {
        if (active) setPublishedExamples(null);
      });
    return () => {
      active = false;
    };
  }, [selectedId, selectedIsDraft]);

  useEffect(() => {
    if (!landscapeId || landscapeId === selectedId) return;
    const model = availableModels.find((item) => item.id === landscapeId);
    if (!model) return;
    setSelectedId(model.id);
    setParameters(model.parameters);
    const nextMaxDepth = Math.max(1, Math.min(4, model.positions.length));
    setMaxDepth(nextMaxDepth);
    setDepthWeights(initialWeights(nextMaxDepth));
    setSampleSize(Math.min(10_000, sampleCapacity(model.positions.length, nextMaxDepth)));
    setPreview(null);
    setError(null);
  }, [availableModels, landscapeId, selectedId]);

  function selectModel(modelId: string) {
    const model = availableModels.find((item) => item.id === modelId);
    if (!model) return;
    setSelectedId(modelId);
    setParameters(model.parameters);
    const nextMaxDepth = Math.max(1, Math.min(4, model.positions.length));
    setMaxDepth(nextMaxDepth);
    setDepthWeights(initialWeights(nextMaxDepth));
    setSampleSize(Math.min(10_000, sampleCapacity(model.positions.length, nextMaxDepth)));
    setPreview(null);
    setError(null);
  }

  function changeMaxDepth(value: number) {
    const positionCount = selectedModel?.positions.length ?? 1;
    const bounded = Math.max(1, Math.min(positionCount, Math.floor(value || 1)));
    setMaxDepth(bounded);
    setSampleSize((current) => Math.min(current, sampleCapacity(positionCount, bounded)));
    setDepthWeights((current) => {
      const next: Record<number, number> = {};
      for (let depth = 1; depth <= bounded; depth++) next[depth] = current[depth] ?? 1;
      return next;
    });
  }

  async function applyPreview(): Promise<boolean> {
    if (!selectedModel) return false;
    setRunning(true);
    setError(null);
    try {
      const source = sourceFromModel(selectedModel, draft?.id);
      const result = await runPreview(source, parameters, sampleSize, depthWeights);
      setPreview(result);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The preview failed.");
      return false;
    } finally {
      setRunning(false);
    }
  }

  async function createLandscape() {
    if (!selectedModel) return;
    if (await applyPreview()) onExploreLandscape?.(selectedModel.id);
  }

  function acceptDraft(created: DraftResponse) {
    const summary = created.summary;
    const model: ModelSummary = {
      id: created.id,
      name: "Custom draft",
      kind: "custom",
      description: "Temporary single-site DMS benchmark draft.",
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
      },
    };
    setDraft(created);
    setDraftModel(model);
    setSelectedId(created.id);
    setParameters(DEFAULT_PARAMETERS);
    const nextMaxDepth = Math.max(1, Math.min(4, summary.positionCount));
    setMaxDepth(nextMaxDepth);
    setDepthWeights(initialWeights(nextMaxDepth));
    setSampleSize(Math.min(10_000, sampleCapacity(summary.positionCount, nextMaxDepth)));
    setPreview(null);
    setError(null);
    onPageChange?.("studio");
  }

  async function publish(name: string) {
    if (!draft) return;
    const model = await publishModel(draft.id, name, parameters);
    setPublishedExamples(model.examples);
    await refreshModels();
    setSelectedId(model.id);
    setDraft(null);
    setDraftModel(null);
    onExploreLandscape?.(model.id);
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
          selectModel(modelId);
          onExploreLandscape?.(modelId);
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
              running={running}
              createLandscape={createLandscape}
              acceptDraft={acceptDraft}
            />
          )}
          {page === "explore" && (
            <LandscapeExplorer
              selectedModel={selectedModel}
              parameters={parameters}
              sampleSize={sampleSize}
              sampleLimit={sampleCapacity(selectedModel?.positions.length ?? 1, maxDepth)}
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
                selectModel(modelId);
                onExploreLandscape?.(modelId);
              }}
            />
          )}
          {page === "leaderboards" && <Leaderboards models={models} />}
        </main>
      </section>
    </Sidebar.Provider>
  );
}

function AppSidebar({
  page,
  models,
  selectedId,
  loading,
  onPageChange,
  onSelectLandscape,
}: {
  page: Page;
  models: ModelSummary[];
  selectedId: string;
  loading: boolean;
  onPageChange?: (page: Page) => void;
  onSelectLandscape: (modelId: string) => void;
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
  );
}

interface LandscapeBuilderProps {
  models: ModelSummary[];
  selectedModel: ModelSummary | null;
  loadingModels: boolean;
  selectModel: (id: string) => void;
  parameters: ModelParameters;
  setParameters: (value: ModelParameters) => void;
  parametersLocked: boolean;
  hasStructure: boolean;
  canFitNonlinearity: boolean;
  running: boolean;
  createLandscape: () => Promise<void>;
  acceptDraft: (draft: DraftResponse) => void;
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
    running,
    createLandscape,
    acceptDraft,
  } = props;
  const [sourceMode, setSourceMode] = useState<"preset" | "upload">("preset");
  const modelItems = Object.fromEntries(models.map((model) => [model.id, model.name]));

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
                acceptDraft(created);
                setSourceMode("preset");
              }}
            />
          )}
        </LayerCard>

        <ParameterPanel
          value={parameters}
          onChange={setParameters}
          locked={parametersLocked}
          hasStructure={hasStructure}
          canFitNonlinearity={canFitNonlinearity}
        />
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
  );
}

interface SamplingProps {
  selectedModel: ModelSummary | null;
  sampleSize: number;
  sampleLimit: number;
  setSampleSize: (value: number) => void;
  maxDepth: number;
  changeMaxDepth: (value: number) => void;
  depthWeights: Record<number, number>;
  setDepthWeights: (
    update: Record<number, number> | ((current: Record<number, number>) => Record<number, number>),
  ) => void;
  running: boolean;
  applyPreview: () => Promise<boolean>;
}

function LandscapeExplorer({
  selectedModel,
  parameters,
  preview,
  draft,
  publish,
  publishedExamples,
  onBack,
  ...sampling
}: SamplingProps & {
  parameters: ModelParameters;
  preview: PreviewResponse | null;
  draft: DraftResponse | null;
  publish: (name: string) => Promise<void>;
  publishedExamples: CodeExamples | null;
  onBack: () => void;
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
        selectedModel={selectedModel}
        draft={draft}
        parameters={parameters}
      />
      <SamplePanel {...sampling} selectedModel={selectedModel} hasPreview={preview !== null} />

      {preview ? (
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
  );
}

function mutationExample(model: ModelSummary | null): string {
  if (!model) return "V39A+D40G";
  return model.positions
    .slice(0, 2)
    .map((position, index) => {
      const wildType = model.wtCompact[index];
      const mutant = wildType === "A" ? "G" : "A";
      return `${wildType}${position}${mutant}`;
    })
    .join("+");
}

function ExactQueryPanel({
  selectedModel,
  draft,
  parameters,
}: {
  selectedModel: ModelSummary | null;
  draft: DraftResponse | null;
  parameters: ModelParameters;
}) {
  const [mutation, setMutation] = useState("");
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [querying, setQuerying] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  useEffect(() => {
    setMutation("");
    setResult(null);
    setQueryError(null);
  }, [selectedModel?.id, parameters]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!selectedModel || !mutation.trim()) return;
    setQuerying(true);
    setQueryError(null);
    try {
      const source = sourceFromModel(selectedModel, draft?.id);
      setResult(await evaluateLandscape(source, parameters, mutation.trim().toUpperCase()));
    } catch (cause) {
      setResult(null);
      setQueryError(cause instanceof Error ? cause.message : "Could not score this mutation.");
    } finally {
      setQuerying(false);
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
    </LayerCard>
  );
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
          const depth = Number(rawDepth);
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
          );
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
  );
}

function WorkflowProgress({ active }: { active: "create" | "explore" }) {
  return (
    <div className="workflow-progress" aria-label="Landscape workflow">
      <div className={active === "create" ? "workflow-stage active" : "workflow-stage done"}>
        <span>1</span>
        Create landscape
      </div>
      <div className="workflow-connector" />
      <div className={active === "explore" ? "workflow-stage active" : "workflow-stage"}>
        <span>2</span>
        Explore landscape
      </div>
    </div>
  );
}

interface ParameterPanelProps {
  value: ModelParameters;
  onChange: (value: ModelParameters) => void;
  locked: boolean;
  hasStructure: boolean;
  canFitNonlinearity: boolean;
}

function ParameterPanel({
  value,
  onChange,
  locked,
  hasStructure,
  canFitNonlinearity,
}: ParameterPanelProps) {
  function set<K extends keyof ModelParameters>(key: K, next: ModelParameters[K]) {
    onChange({ ...value, [key]: next });
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
  );
}

function PreviewResults({
  preview,
  draft,
  publish,
  examples,
}: {
  preview: PreviewResponse;
  draft: DraftResponse | null;
  publish: (name: string) => Promise<void>;
  examples: CodeExamples | null;
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
  );
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
  );
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
  );
}

function PublishPanel({ onPublish }: { onPublish: (name: string) => Promise<void> }) {
  const [name, setName] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [publishError, setPublishError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setPublishing(true);
    setPublishError(null);
    try {
      await onPublish(name.trim());
    } catch (cause) {
      setPublishError(cause instanceof Error ? cause.message : "Publishing failed.");
    } finally {
      setPublishing(false);
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
  );
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
          <ClipboardText
            size="sm"
            text="Copy local Python"
            textToCopy={examples.localPython}
          />
        </div>
        <div>
          <Text bold>Hosted API</Text>
          <pre className="code-preview">{examples.hostedPython}</pre>
          <ClipboardText
            size="sm"
            text="Copy hosted Python"
            textToCopy={examples.hostedPython}
          />
        </div>
      </Grid>
    </LayerCard>
  );
}

function CustomUploadPanel({ onCreated }: { onCreated: (draft: DraftResponse) => void }) {
  type StructureMode = "none" | "upload" | "esmfold";
  const [dmsCsv, setDmsCsv] = useState("");
  const [dmsName, setDmsName] = useState("");
  const [wtFasta, setWtFasta] = useState("");
  const [fastaName, setFastaName] = useState("");
  const [structureMode, setStructureMode] = useState<StructureMode>("none");
  const [structureText, setStructureText] = useState("");
  const [structureName, setStructureName] = useState("");
  const [structureFormat, setStructureFormat] = useState<"pdb" | "cif" | "mmcif" | undefined>();
  const [creating, setCreating] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  async function readFile(
    file: File,
    setter: (text: string) => void,
    setName: (name: string) => void,
  ) {
    setter(await file.text());
    setName(file.name);
  }

  async function readStructure(file: File) {
    const extension = file.name.split(".").pop()?.toLowerCase();
    setStructureFormat(extension === "cif" || extension === "mmcif" ? extension : "pdb");
    await readFile(file, setStructureText, setStructureName);
  }

  function chooseStructureMode(value: StructureMode) {
    setStructureMode(value);
    if (value !== "upload") {
      setStructureText("");
      setStructureName("");
      setStructureFormat(undefined);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setCreating(true);
    setUploadError(null);
    try {
      const next = await createDraft({
        dmsCsv,
        wtFasta,
        structureText:
          structureMode === "upload" ? structureText || undefined : undefined,
        structureFormat: structureMode === "upload" ? structureFormat : undefined,
        structureInference: structureMode === "esmfold" ? "esmfold" : "none",
      });
      onCreated(next);
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "The upload failed.");
    } finally {
      setCreating(false);
    }
  }

  return (
    <form className="upload-form source-upload-form" onSubmit={(event) => void submit(event)}>
      {uploadError && <Banner variant="error" title="Upload failed" description={uploadError} />}
      <div className="drop-zone-grid">
        <DropFileField
          label="Single-site DMS CSV"
          description="position, wildtype, mutant, fitness"
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
  );
}

function DropFileField({
  label,
  description,
  required = false,
  accept,
  fileName,
  onFile,
}: {
  label: string;
  description: string;
  required?: boolean;
  accept: string;
  fileName: string;
  onFile: (file: File) => void | Promise<void>;
}) {
  const [dragging, setDragging] = useState(false);

  function acceptDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void onFile(file);
  }

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    if (file) void onFile(file);
  }

  return (
    <label
      className={dragging ? "drop-file-field dragging" : "drop-file-field"}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={acceptDrop}
    >
      <input type="file" accept={accept} onChange={selectFile} aria-required={required} />
      <UploadSimple size={22} weight={fileName ? "fill" : "regular"} />
      <span className="drop-file-copy">
        <strong>{label}</strong>
        <small>{fileName || `Drop file here or browse · ${description}`}</small>
      </span>
      {fileName && <CheckCircle className="success-icon" size={18} weight="fill" />}
    </label>
  );
}

function ModelCatalogue({
  models,
  loading,
  onRefresh,
  onExplore,
}: {
  models: ModelSummary[];
  loading: boolean;
  onRefresh: () => Promise<void>;
  onExplore: (modelId: string) => void;
}) {
  return (
    <div className="studio-stack">
      <section className="page-heading">
        <div>
          <Text variant="heading2" as="h1">
            Landscape library
          </Text>
          <Text variant="secondary">
            Frozen references and saved custom benchmark landscapes.
          </Text>
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
  );
}

export function Leaderboards({ models }: { models: ModelSummary[] }) {
  const [boards, setBoards] = useState<Leaderboard[]>([]);
  const [selectedBoardId, setSelectedBoardId] = useState("");
  const [loading, setLoading] = useState(true);
  const [boardError, setBoardError] = useState<string | null>(null);

  async function refreshBoards(preferredId?: string) {
    setLoading(true);
    setBoardError(null);
    try {
      const next = await listLeaderboards();
      setBoards(next);
      setSelectedBoardId((current) =>
        preferredId && next.some((board) => board.id === preferredId)
          ? preferredId
          : next.some((board) => board.id === current)
            ? current
            : (next[0]?.id ?? ""),
      );
    } catch (cause) {
      setBoardError(cause instanceof Error ? cause.message : "Could not load leaderboards.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refreshBoards();
  }, []);

  const selectedBoard = boards.find((board) => board.id === selectedBoardId) ?? null;
  const landscapeName =
    models.find((model) => model.id === selectedBoard?.landscapeId)?.name ??
    selectedBoard?.landscapeId;

  async function makeBoard(name: string, landscapeId: string, queryBudget: number) {
    const board = await createLeaderboard(name, landscapeId, queryBudget);
    await refreshBoards(board.id);
  }

  async function addResult(modelName: string, successes: number, trials: number) {
    if (!selectedBoard) return;
    await addLeaderboardEntry(selectedBoard.id, modelName, successes, trials);
    await refreshBoards(selectedBoard.id);
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
        <Banner
          variant="error"
          title="Leaderboard request failed"
          description={boardError}
        />
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
                      {landscapeName} · {selectedBoard.queryBudget.toLocaleString()} queries
                      per run
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
  );
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
  };
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
  );
}

function CreateLeaderboardDialog({
  models,
  onCreate,
}: {
  models: ModelSummary[];
  onCreate: (name: string, landscapeId: string, queryBudget: number) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [landscapeId, setLandscapeId] = useState(models[0]?.id ?? "");
  const [queryBudget, setQueryBudget] = useState(1_000);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const items = Object.fromEntries(models.map((model) => [model.id, model.name]));

  useEffect(() => {
    if (!landscapeId && models[0]) setLandscapeId(models[0].id);
  }, [landscapeId, models]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      await onCreate(name.trim(), landscapeId, queryBudget);
      setName("");
      setOpen(false);
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not create leaderboard.");
    } finally {
      setSaving(false);
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
  );
}

function AddResultDialog({
  onAdd,
}: {
  onAdd: (modelName: string, successes: number, trials: number) => Promise<void>;
}) {
  const [modelName, setModelName] = useState("");
  const [successes, setSuccesses] = useState(0);
  const [trials, setTrials] = useState(10);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      await onAdd(modelName.trim(), successes, trials);
      setModelName("");
      setOpen(false);
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not add result.");
    } finally {
      setSaving(false);
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
  );
}

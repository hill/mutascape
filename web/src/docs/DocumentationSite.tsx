import { ArrowRight, Check, Copy } from "@phosphor-icons/react"
import { useEffect, useMemo, useState } from "react"
import generatedDocs from "../../data/docs.generated.json"
import { highlightPython } from "../python-highlight"
import "./docs.css"
import { Equation, InlineMath } from "./Equation"

export const DOCS_PAGES = [
  { id: "docs", title: "Introduction", path: "/docs" },
  { id: "docs-inputs", title: "Input data", path: "/docs/inputs" },
  { id: "docs-model", title: "The fitness model", path: "/docs/model" },
  { id: "docs-benchmarking", title: "Benchmarking", path: "/docs/benchmarking" },
  { id: "docs-api", title: "API reference", path: "/docs/api" },
] as const

export type DocsPage = (typeof DOCS_PAGES)[number]["id"]

const DOCS_PAGE_IDS = new Set<string>(DOCS_PAGES.map((entry) => entry.id))

export function isDocsPage(page: string): page is DocsPage {
  return DOCS_PAGE_IDS.has(page)
}

interface DocumentationSiteProps {
  page: DocsPage
  onNavigate: (page: DocsPage) => void
  onOpenStudio: () => void
}

interface Section {
  id: string
  title: string
}

const quickStartCode = `from mutascape import from_dms

generator = from_dms(
    "data/creilov/creilov.csv",
    fasta="data/creilov/creilov_wt.fasta",
    maximum_depth=4,
)
oracle = generator.draw(seed=42)

print(generator.domain)
print(oracle.score_many([
    "WT", "A29H", "R60D", "A29H+R60D"
]))`

const campaignCode = `from mutascape import Campaign, run_campaign

def random_search(campaign: Campaign) -> None:
    while campaign.remaining:
        size = min(campaign.batch_size, campaign.remaining)
        batch = campaign.sample(size, seed=100 + campaign.round)
        campaign.measure(batch)

candidates = [
    variant
    for depth in range(1, 5)
    for variant in generator.domain.sample(
        depth=depth, n=250, seed=20 + depth
    )
]
result = run_campaign(
    random_search,
    oracle,
    candidates,
    budget=192,
    batch_size=48,
    seed=7,
)
print(result.to_dict())`

const positionSchema = `position,wildtype,mutant,fitness
39,V,A,-1.82
39,V,C,-0.44
40,D,A,0.31`

const variantSchema = `variant,fitness
WT,0.0
V39A,-1.82
V39A+D40G,-1.14`

const mavedbSchema = `accession,hgvs_pro,score
urn:mavedb:00001222-b-2#1,p.Ser1Ala,-0.42
urn:mavedb:00001222-b-2#2,p.Ser1Ter,-3.10
urn:mavedb:00001222-b-2#3,p.Asp2Gly,0.08`

function oracleOutput(): string {
  const rows = generatedDocs.oracle.labels.map(
    (label, index) => `${label.padEnd(12)} ${generatedDocs.oracle.scores[index].toFixed(4)}`,
  )
  return [generatedDocs.oracle.domain, ...rows].join("\n")
}

function campaignOutput(): string {
  const result = generatedDocs.campaign
  return [
    `{`,
    `  "best_found": ${result.bestFound.toFixed(4)},`,
    `  "optimum": ${result.optimum.toFixed(4)},`,
    `  "simple_regret": ${result.simpleRegret.toFixed(4)},`,
    `  "queries": ${result.queries},`,
    `  "rounds": ${result.rounds}`,
    `}`,
  ].join("\n")
}

export default function DocumentationSite({
  page,
  onNavigate,
  onOpenStudio,
}: DocumentationSiteProps) {
  const title = DOCS_PAGES.find((entry) => entry.id === page)?.title ?? "Documentation"

  useEffect(() => {
    document.title = `${title} | mutascape`
    return () => {
      document.title = "mutascape"
    }
  }, [title])

  const [document_, sections] = useMemo(
    () => renderPage(page, onNavigate, onOpenStudio),
    [page, onNavigate, onOpenStudio],
  )

  return (
    <div className="docs-site">
      <div className="docs-layout">
        <DocsContents page={page} onNavigate={onNavigate} />
        {document_}
        <DocsOnThisPage sections={sections} />
      </div>
    </div>
  )
}

function renderPage(
  page: DocsPage,
  onNavigate: (page: DocsPage) => void,
  onOpenStudio: () => void,
): [React.ReactNode, Section[]] {
  switch (page) {
    case "docs-inputs":
      return [<InputsPage key={page} onNavigate={onNavigate} />, INPUT_SECTIONS]
    case "docs-model":
      return [<ModelPage key={page} onNavigate={onNavigate} />, MODEL_SECTIONS]
    case "docs-benchmarking":
      return [<BenchmarkingPage key={page} onNavigate={onNavigate} />, BENCHMARK_SECTIONS]
    case "docs-api":
      return [<ApiPage key={page} onOpenStudio={onOpenStudio} />, apiSections()]
    default:
      return [<OverviewPage key={page} onNavigate={onNavigate} />, OVERVIEW_SECTIONS]
  }
}

function DocsContents({
  page,
  onNavigate,
}: {
  page: DocsPage
  onNavigate: (page: DocsPage) => void
}) {
  return (
    <aside className="docs-contents" aria-label="Documentation">
      <p>Contents</p>
      <nav>
        {DOCS_PAGES.map((entry) => (
          <button
            className={entry.id === page ? "active" : undefined}
            key={entry.id}
            onClick={() => onNavigate(entry.id)}
            type="button"
          >
            {entry.title}
          </button>
        ))}
      </nav>
    </aside>
  )
}

function DocsOnThisPage({ sections }: { sections: Section[] }) {
  return (
    <aside className="docs-toc" aria-label="On this page">
      <p>On this page</p>
      <nav>
        {sections.map((section) => (
          <a href={`#${section.id}`} key={section.id}>
            {section.title}
          </a>
        ))}
      </nav>
    </aside>
  )
}

/* Introduction ---------------------------------------------------------- */

const OVERVIEW_SECTIONS: Section[] = [
  { id: "what-mutascape-does", title: "What Mutascape does" },
  { id: "quick-start", title: "Quick start" },
  { id: "example-output", title: "Example output" },
]

function OverviewPage({ onNavigate }: { onNavigate: (page: DocsPage) => void }) {
  return (
    <article className="docs-document">
      <header className="docs-document-header">
        <p className="docs-breadcrumb">Documentation / Introduction</p>
        <h1>Introduction</h1>
        <p className="docs-lead">
          Mutascape creates fixed synthetic fitness landscapes for ML-assisted directed evolution
          benchmarks.
        </p>
      </header>

      <section className="docs-section" id="what-mutascape-does">
        <h2>What Mutascape does</h2>
        <p>
          Give Mutascape a wild-type sequence and measured variants. It returns an oracle for the
          mutation set you choose.
        </p>
        <p>
          Each seed fixes one oracle. Every method can then search the same hidden landscape under
          the same query budget.
        </p>
        <dl className="docs-concept-grid">
          <div>
            <dt>Domain</dt>
            <dd>The variants that a method may query.</dd>
          </div>
          <div>
            <dt>Generator</dt>
            <dd>The model and prior that draw an oracle.</dd>
          </div>
          <div>
            <dt>Oracle</dt>
            <dd>A fixed fitness function hidden from the search method.</dd>
          </div>
        </dl>
        <p>
          The DMS-pairwise generator does not need a protein structure. Its default oracle does not
          read PDB files or ESMFold predictions.
        </p>
        <p className="docs-note">
          Mutascape is a benchmark generator. It does not predict the true fitness of an unmeasured
          protein.
        </p>
      </section>

      <section className="docs-section" id="quick-start">
        <div className="docs-section-heading">
          <h2>Quick start</h2>
          <p>Load DMS data, draw an oracle, and score variants.</p>
        </div>
        <div className="docs-code-stack">
          <CodeBlock label="Python" code={quickStartCode} />
          <CodeBlock label="Output" code={oracleOutput()} output />
        </div>
      </section>

      <FigureSection
        id="example-output"
        title="Example output"
        description="Each group contains 160 CreiLOV variants. Grey shows the single-effect model. Green shows one fixed oracle draw."
        src="/assets/docs/oracle-profile.svg"
        alt="Fitness distributions for a single-effect baseline and fixed synthetic oracle at mutation depths one to four"
      />

      <NextPageLink
        label="Next guide"
        title="Prepare input data"
        onClick={() => onNavigate("docs-inputs")}
      />
    </article>
  )
}

/* Input data ------------------------------------------------------------ */

const INPUT_SECTIONS: Section[] = [
  { id: "what-a-scan-measures", title: "What a scan measures" },
  { id: "what-mutascape-needs", title: "What Mutascape needs" },
  { id: "csv-layouts", title: "CSV layouts" },
  { id: "mavedb", title: "MaveDB" },
  { id: "coverage", title: "Supported mutation domain" },
]

function InputsPage({ onNavigate }: { onNavigate: (page: DocsPage) => void }) {
  return (
    <article className="docs-document">
      <header className="docs-document-header">
        <p className="docs-breadcrumb">Documentation / Input data</p>
        <h1>Prepare input data</h1>
        <p className="docs-lead">
          Mutascape builds every landscape from one wild-type sequence and one table of measured
          variants.
        </p>
      </header>

      <section className="docs-section" id="what-a-scan-measures">
        <h2>What a scan measures</h2>
        <p>
          A deep mutational scan builds a library where each member carries one substitution, then
          puts the whole library through a selection. Growth, binding and fluorescence are all
          common. Sequencing counts each variant before and after.
        </p>
        <p>
          The enrichment of a variant, its count after selection over its count before, becomes its
          functional score. A multiplexed assay of variant effect, or MAVE, is the same idea over
          any readout.
        </p>
        <p>
          One scan therefore gives you the effect of nearly every single substitution, and almost
          nothing about combinations. Mutascape exists to fill in the combinations.
        </p>
      </section>

      <section className="docs-section" id="what-mutascape-needs">
        <h2>What Mutascape needs</h2>
        <dl className="docs-concept-grid">
          <div>
            <dt>Wild-type sequence</dt>
            <dd>A FASTA file or a plain string. Every position index is read against it.</dd>
          </div>
          <div>
            <dt>Functional scores</dt>
            <dd>One row per measured variant, relative to wild type.</dd>
          </div>
        </dl>
        <p className="docs-note">
          Wild type sits at 0.0. A score of -1.0 is a variant one unit worse than the parent, and
          the sign convention of your assay has to match before you upload.
        </p>
      </section>

      <section className="docs-section" id="csv-layouts">
        <div className="docs-section-heading">
          <h2>CSV layouts</h2>
          <p>
            <InlineCode>from_dms</InlineCode> and the studio both accept three layouts.
          </p>
        </div>
        <div className="docs-code-stack">
          <CodeBlock label="Position columns" code={positionSchema} />
          <CodeBlock label="Variant labels" code={variantSchema} />
        </div>
        <p>
          The variant layout carries combinations as well, joined with <InlineCode>+</InlineCode>.
          Any row it holds beyond the singles conditions the landscape directly rather than being
          predicted.
        </p>
        <p>
          A third layout takes a <InlineCode>mutations</InlineCode> column of mapping objects. The
          studio also lets you map arbitrary column names onto any of the three, so an export you
          did not write still uploads.
        </p>
      </section>

      <section className="docs-section" id="mavedb">
        <div className="docs-section-heading">
          <h2>MaveDB</h2>
          <p>Paste a score-set URN or its URL into the studio and it fetches the scores.</p>
        </div>
        <CodeBlock label="MaveDB score set" code={mavedbSchema} />
        <p>
          MaveDB writes variants as HGVS protein notation with three-letter residues, so{" "}
          <InlineCode>p.Ser1Ala</InlineCode> is S1A. Mutascape translates the notation and drops
          synonymous and nonsense rows, since <InlineCode>p.Ser1Ter</InlineCode> is a truncation
          rather than a substitution.
        </p>
        <p className="docs-note">
          MaveDB positions are numbered against the target sequence the depositor registered. Give
          Mutascape that same sequence, not the full-length protein, or every position will land on
          the wrong residue.
        </p>
      </section>

      <section className="docs-section" id="coverage">
        <h2>Supported mutation domain</h2>
        <p>
          A scan rarely covers all 19 substitutions at every position. The oracle domain contains
          each measured single substitution and the combinations that those substitutions define.
        </p>
        <p>
          By default, Mutascape rejects queries with unmeasured substitutions. Optional ESM-2
          completion fills missing singles across the protein. It calibrates masked model scores on
          measured singles and bounds estimates by their observed range. The readout still uses
          measured singles only. Imputed effects are estimates, not experimental observations.
        </p>
        <CodeBlock
          label="Optional whole-protein completion"
          code={`generator = from_dms(
    "creilov.csv", fasta="creilov_wt.fasta", complete="esm2",
)
oracle = generator.draw(seed=42)
oracle["A2N"]                 # calibrated single estimate
oracle.provenance("A2N")      # "imputed"
oracle["A2N+A29H"]            # synthetic combination
oracle.save_manifest("oracle.json")`}
        />
        <p>
          The first run loads ESM-2 weights. The studio upload form offers the same completion
          option. Saved manifests include the imputed values, so restoration needs no model weights.
          CreiLOV validation with hidden positions reduced RMSE from 0.304 to 0.256. Position means
          had lower error when other substitutions at the same position remained measured.
        </p>
      </section>

      <NextPageLink
        label="Next guide"
        title="How the fitness model works"
        onClick={() => onNavigate("docs-model")}
      />
    </article>
  )
}

/* The fitness model ----------------------------------------------------- */

const MODEL_SECTIONS: Section[] = [
  { id: "the-equation", title: "The equation" },
  { id: "singles", title: "Measured anchors" },
  { id: "couplings", title: "Pair interactions" },
  { id: "global-epistasis", title: "Saturating readout" },
]

function ModelPage({ onNavigate }: { onNavigate: (page: DocsPage) => void }) {
  return (
    <article className="docs-document">
      <header className="docs-document-header">
        <p className="docs-breadcrumb">Documentation / The fitness model</p>
        <h1>The fitness model</h1>
        <p className="docs-lead">
          One equation decides every score an oracle returns. Reading it explains most of what the
          package does.
        </p>
      </header>

      <section className="docs-section" id="the-equation">
        <h2>The equation</h2>
        <Equation formula="f_s(x) \;=\; f_{\mathrm{WT}} \;+\; g\!\left( \sum_i \tilde{h}_i(x_i) \;+\; \lambda \sigma \sum_{i<j} \tilde{J}^{(s)}_{ij}(x_i,x_j) \right)" />
        <dl className="docs-concept-grid">
          <div>
            <dt>
              <InlineMath formula="\tilde{h}_i" />
            </dt>
            <dd>The measured single effect in the latent readout space.</dd>
          </div>
          <div>
            <dt>
              <InlineMath formula="\tilde{J}^{(s)}_{ij}" />
            </dt>
            <dd>A seed-specific pair coefficient with a zero wild-type baseline.</dd>
          </div>
          <div>
            <dt>
              <InlineMath formula="\lambda" />
            </dt>
            <dd>The pair strength. The validated default is 0.1.</dd>
          </div>
          <div>
            <dt>
              <InlineMath formula="\sigma" />
            </dt>
            <dd>The scale of the target protein's measured single effects.</dd>
          </div>
          <div>
            <dt>
              <InlineMath formula="g" />
            </dt>
            <dd>The global-epistasis function, a saturating sigmoid.</dd>
          </div>
        </dl>
        <p className="docs-note">
          The model uses measured DMS effects, random pair interactions, and an empirical readout.
          It does not use a contact map or claim a structural mechanism.
        </p>
      </section>

      <section className="docs-section" id="singles">
        <h2>Measured anchors</h2>
        <p>
          Additive effects are summed in the preimage of <InlineMath formula="g" />, so each stored
          effect is the value that <InlineMath formula="g" /> maps back onto the measurement.
        </p>
        <Equation formula="\tilde{h}_i \;=\; g^{-1}(h_i) \qquad\Longrightarrow\qquad f(\text{single}_i) \;=\; g\!\left(g^{-1}(h_i)\right) \;=\; h_i" />
        <p>
          Wild type, single mutants, and any supplied combinations return their measured scores. The
          model generates values only for combinations absent from the input.
        </p>
      </section>

      <section className="docs-section" id="couplings">
        <h2>Pair interactions</h2>
        <p>
          Each seed draws a stable table for every pair of mutable positions. A double difference
          removes the wild-type row and column.
        </p>
        <Equation formula="J_{ij}^{\ast} \;=\; J(m_i, m_j) \;-\; J(w_i, m_j) \;-\; J(m_i, w_j) \;+\; J(w_i, w_j)" />
        <p>
          Here <InlineMath formula="m" /> is the mutant residue and <InlineMath formula="w" /> the
          wild-type one. The correction is zero whenever either position is unmutated, so singles
          pick up no coupling and the sum can run over mutated pairs alone.
        </p>
        <p className="docs-note">
          The same seed returns the same pair table and the same score for every query. Changing the
          seed draws another benchmark landscape from the generator.
        </p>
      </section>

      <section className="docs-section" id="global-epistasis">
        <h2>Saturating readout</h2>
        <p>
          Real assays saturate. A protein cannot fold better than folded, and a dead variant cannot
          get deader once the readout hits its floor.
        </p>
        <Equation formula="g(z) \;=\; y_{\min} \;+\; \frac{y_{\max} - y_{\min}}{1 + e^{-k(z - z_0)}}" />
        <p>
          Target singles set the local scale. Other measured combinatorial landscapes supply the
          typical assay range. Their pooled values set the lower and upper bounds of the readout.
        </p>
      </section>

      <NextPageLink
        label="Next guide"
        title="Run a benchmark campaign"
        onClick={() => onNavigate("docs-benchmarking")}
      />
    </article>
  )
}

/* Benchmarking ---------------------------------------------------------- */

const BENCHMARK_SECTIONS: Section[] = [
  { id: "protocol", title: "Recommended protocol" },
  { id: "creilov-shape", title: "CreiLOV shape check" },
  { id: "minimal-campaign", title: "Minimal campaign" },
  { id: "campaign-output", title: "Example output" },
  { id: "interpretation", title: "Interpretation" },
]

function BenchmarkingPage({ onNavigate }: { onNavigate: (page: DocsPage) => void }) {
  return (
    <article className="docs-document">
      <header className="docs-document-header">
        <p className="docs-breadcrumb">Documentation / Benchmarking</p>
        <h1>Benchmark under an assay budget</h1>
        <p className="docs-lead">
          A campaign hides fitness until a method spends a query. It records the best measured
          variant at the end.
        </p>
      </header>

      <section className="docs-section" id="protocol">
        <div className="docs-section-heading">
          <h2>Recommended protocol</h2>
          <p>Give every method the same task and query budget.</p>
        </div>
        <ol className="docs-protocol-list">
          <ProtocolItem
            number="01"
            title="Fix the oracle draw"
            text="Save the generator version, source digest, coupling, and draw seed."
          />
          <ProtocolItem
            number="02"
            title="Match the search space"
            text="Give every method the same candidate domain and sequence encoding."
          />
          <ProtocolItem
            number="03"
            title="Use a relevant budget"
            text="Match the query count and batch size to the intended engineering campaign."
          />
          <ProtocolItem
            number="04"
            title="Repeat methods and oracle draws"
            text="Run ten search trials on three matched oracle draws."
          />
        </ol>
      </section>

      <FigureSection
        id="creilov-shape"
        title="CreiLOV shape check"
        description="Thirty fixed draws match the standardised histogram shape and the measured allocation of variance across Walsh interaction orders. Raw fitness location remains a known mismatch."
        src="/assets/docs/creilov-shape-validation.png"
        alt="Measured and synthetic CreiLOV fitness distributions and Walsh interaction-order spectra"
      />

      <section className="docs-section" id="minimal-campaign">
        <h2>Adversarial assessment</h2>
        <p>
          A classifier tries to distinguish measured and synthetic fitness on held-out genotypes.
          AUC and balanced accuracy near 0.5 mean weak discrimination for that classifier. They do
          not prove equivalence. Each real and synthetic genotype pair stays in one fold.
          Permutation tests swap source labels within pairs and refit the classifier.
        </p>
        <p>
          The CreiLOV assay-scale test gives AUC above 0.999 for three synthetic draws. This exposes
          the raw fitness mismatch despite the close standardised histogram. Replicates 1 and 2 give
          AUC near 0.51. Comparisons with replicate 3 give about 0.66. Replicate disagreement is a
          reference, not a universal noise floor.
        </p>
        <p>
          The original DMS and combinatorial assay disagree even on shared single mutants.
          Same-assay singles reduce the offset, but do not recover the negative trend from combining
          mutations. Pilot-calibrated models remain experimental and need measured combinations.
          They do not replace the singles-only generator described here.
        </p>
        <CodeBlock
          label="Paired classifier test"
          code={`import numpy as np
from mutascape.benchmarks import classifier_two_sample_test

# Each row describes the same genotype in both arrays.
groups = np.arange(len(real_features))
result = classifier_two_sample_test(
    real_features, synthetic_features,
    real_groups=groups, synthetic_groups=groups,
    classifier="histogram", n_permutations=99, seed=42,
)
print(result.auc, result.accuracy)`}
        />
        <FigureSection
          id="creilov-c2st"
          title="CreiLOV classifier check"
          description="Points show median AUC across three genotype samples. Bars show the sample range, not confidence intervals."
          src="/assets/docs/creilov-c2st.svg"
          alt="Replicate discrimination near 0.51 to 0.67 and synthetic discrimination near one"
        />
        <div className="docs-section-heading">
          <h2>Minimal campaign</h2>
          <p>
            The method sees candidates, measured variants, and its remaining budget.
            <InlineCode>campaign.measure</InlineCode> reveals fitness.
          </p>
        </div>
        <dl className="docs-facts">
          <div>
            <dt>Candidate pool</dt>
            <dd>{generatedDocs.campaign.candidateCount.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Budget</dt>
            <dd>{generatedDocs.campaign.budget} queries</dd>
          </div>
          <div>
            <dt>Rounds</dt>
            <dd>{generatedDocs.campaign.rounds}</dd>
          </div>
        </dl>
        <div className="docs-code-stack">
          <CodeBlock label="Python" code={campaignCode} />
          <CodeBlock label="Output" code={campaignOutput()} output />
        </div>
      </section>

      <FigureSection
        id="campaign-output"
        title="Example output"
        description="The green trace shows the best measured variant across 192 queries. Vertical guides mark each 48-query round."
        src="/assets/docs/campaign-trace.svg"
        alt="Best fitness found by random search over 192 queries, with the best candidate shown as a reference"
      />

      <section className="docs-section" id="interpretation">
        <div className="docs-section-heading">
          <h2>Interpretation</h2>
          <p>Limit each claim to what the benchmark measures.</p>
        </div>
        <div className="docs-claim-grid">
          <article className="docs-claim-card supported">
            <span className="docs-claim-label">
              <Check weight="bold" /> Supported use
            </span>
            <h3>Compare search decisions</h3>
            <p>Report method rankings, pairwise winners, regret, and query counts.</p>
          </article>
          <article className="docs-claim-card restricted">
            <span className="docs-claim-label">Current boundary</span>
            <h3>Do not infer molecular truth</h3>
            <p>
              Do not claim exact genotype fitness, peak counts, adaptive paths, or molecular
              epistasis.
            </p>
          </article>
        </div>
      </section>

      <NextPageLink
        label="Next guide"
        title="API reference"
        onClick={() => onNavigate("docs-api")}
      />
    </article>
  )
}

/* API reference --------------------------------------------------------- */

function sectionId(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-")
}

function apiSections(): Section[] {
  return generatedDocs.api.map((group) => ({ id: sectionId(group.title), title: group.title }))
}

function ApiPage({ onOpenStudio }: { onOpenStudio: () => void }) {
  return (
    <article className="docs-document">
      <header className="docs-document-header">
        <p className="docs-breadcrumb">Documentation / API reference</p>
        <h1>API reference</h1>
        <p className="docs-lead">
          Everything below is exported from the <InlineCode>mutascape</InlineCode> top-level
          package.
        </p>
      </header>

      {generatedDocs.api.map((group) => (
        <section className="docs-section" id={sectionId(group.title)} key={group.title}>
          <h2>{group.title}</h2>
          <div className="docs-api-list">
            {group.entries.map((entry) => (
              <ApiEntry entry={entry} key={entry.name} />
            ))}
          </div>
        </section>
      ))}

      <NextPageLink label="Try it" title="Open the landscape studio" onClick={onOpenStudio} />
    </article>
  )
}

type ApiEntryData = (typeof generatedDocs.api)[number]["entries"][number]

function ApiEntry({ entry }: { entry: ApiEntryData }) {
  return (
    <article className="docs-api-entry">
      <h3>{entry.name}</h3>
      <pre className="docs-api-signature">
        <code>{highlightPython(entry.signature)}</code>
      </pre>
      <p>{entry.summary}</p>
      {entry.notes.map((note) =>
        note.kind === "list" ? (
          <ul className="docs-api-notes" key={note.items?.join("|")}>
            {note.items?.map((item) => (
              <li key={item}>
                <Prose text={item} />
              </li>
            ))}
          </ul>
        ) : (
          <p key={note.text}>
            <Prose text={note.text ?? ""} />
          </p>
        ),
      )}
      {entry.parameters.length > 0 && (
        <dl className="docs-api-parameters">
          {entry.parameters.map((parameter) => (
            <div key={parameter.name}>
              <dt>
                {parameter.name}
                {parameter.keywordOnly && <span className="docs-api-flag">keyword</span>}
              </dt>
              <dd>
                <code>{parameter.annotation || "any"}</code>
                {parameter.default && <span>defaults to {parameter.default}</span>}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {entry.returns && (
        <p className="docs-api-returns">
          Returns <code>{entry.returns}</code>
        </p>
      )}
    </article>
  )
}

/* Shared ---------------------------------------------------------------- */

function InlineCode({ children }: { children: string }) {
  return <code>{children}</code>
}

/** Renders single-backtick spans as code, matching the docstring convention. */
function Prose({ text }: { text: string }) {
  return (
    <>
      {text.split("`").map((part, index) =>
        index % 2 === 1 ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: split parts have no stable identity
          <code key={index}>{part}</code>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: split parts have no stable identity
          <span key={index}>{part}</span>
        ),
      )}
    </>
  )
}

function CodeBlock({
  label,
  code,
  output = false,
}: {
  label: string
  code: string
  output?: boolean
}) {
  const [copied, setCopied] = useState(false)

  async function copyCode() {
    await navigator.clipboard?.writeText(code)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  return (
    <div className={output ? "docs-code-block output" : "docs-code-block"}>
      <div className="docs-code-toolbar">
        <span>{label}</span>
        {!output && (
          <button type="button" onClick={() => void copyCode()} aria-label="Copy code">
            {copied ? <Check /> : <Copy />}
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>
      <pre>
        <code className={output ? undefined : "language-python"}>
          {output ? code : highlightPython(code)}
        </code>
      </pre>
    </div>
  )
}

function FigureSection({
  id,
  title,
  description,
  src,
  alt,
}: {
  id: string
  title: string
  description: string
  src: string
  alt: string
}) {
  return (
    <section className="docs-section docs-figure-section" id={id}>
      <div className="docs-section-heading">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      <figure>
        <img src={src} alt={alt} loading="lazy" />
      </figure>
    </section>
  )
}

function ProtocolItem({ number, title, text }: { number: string; title: string; text: string }) {
  return (
    <li>
      <span className="docs-protocol-number">{number}</span>
      <div>
        <h3>{title}</h3>
        <p>{text}</p>
      </div>
    </li>
  )
}

function NextPageLink({
  label,
  title,
  onClick,
}: {
  label: string
  title: string
  onClick: () => void
}) {
  return (
    <button className="docs-next-link" type="button" onClick={onClick}>
      <span className="docs-next-copy">
        <small>{label}</small>
        <strong>{title}</strong>
      </span>
      <ArrowRight />
    </button>
  )
}

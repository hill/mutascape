import { Badge, Button, Input, LayerCard, Select, Tabs, Text } from "@cloudflare/kumo"
import { UploadSimple } from "@phosphor-icons/react"
import { type ChangeEvent, useEffect, useRef, useState } from "react"
import { ApiError, fetchStructure } from "./api"
import { DIVERGING } from "./charts"
import type { PreviewSource, StructureResidue, StructureSource, StructureView } from "./types"

const SOURCE_TABS = [
  { value: "rcsb", label: "PDB ID" },
  { value: "upload", label: "Upload" },
  { value: "esmfold", label: "ESMFold" },
]

const BACKBONE = "#c9ced6"
const UNASSAYED = "#8d949e"

function parseHex(colour: string): [number, number, number] {
  return [
    Number.parseInt(colour.slice(1, 3), 16),
    Number.parseInt(colour.slice(3, 5), 16),
    Number.parseInt(colour.slice(5, 7), 16),
  ]
}

/** Sample the heatmap's diverging ramp so the two views agree on what a colour means. */
export function effectColour(effect: number, bound: number): string {
  const t = Math.min(1, Math.max(0, (effect / bound + 1) / 2)) * (DIVERGING.length - 1)
  const low = Math.floor(t)
  const high = Math.min(DIVERGING.length - 1, low + 1)
  const [r1, g1, b1] = parseHex(DIVERGING[low])
  const [r2, g2, b2] = parseHex(DIVERGING[high])
  const mix = t - low
  const channel = (a: number, b: number) => Math.round(a + (b - a) * mix)
  return `#${[channel(r1, r2), channel(g1, g2), channel(b1, b2)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")}`
}

function residueLabel(residue: StructureResidue): string {
  const effect =
    residue.effect === null ? "no measurement" : `mean effect ${residue.effect.toFixed(2)}`
  return `${residue.parent}${residue.position} (${effect})`
}

function useViewer(view: StructureView | null) {
  const mount = useRef<HTMLDivElement | null>(null)
  const [selected, setSelected] = useState<StructureResidue | null>(null)

  useEffect(() => {
    const element = mount.current
    if (!view || !element) return
    setSelected(null)

    let viewer: { clear: () => void } | null = null
    let live = true

    // 3Dmol is 600 kB of WebGL and only this panel needs it, so it stays out of the entry
    // bundle and loads when a structure arrives.
    void import("3dmol").then((library) => {
      if (!live || !mount.current) return
      const created = library.createViewer(mount.current, { backgroundAlpha: 0 })
      viewer = created
      created.addModel(view.pdb, view.format === "mmcif" ? "cif" : "pdb")
      created.setStyle({}, { cartoon: { color: BACKBONE, opacity: 0.85 } })

      for (const residue of view.residues) {
        const colour =
          residue.effect === null ? UNASSAYED : effectColour(residue.effect, view.bound)
        const selection = { chain: view.chainId, resi: residue.residue }
        created.setStyle(selection, {
          cartoon: { color: colour },
          stick: { color: colour, radius: 0.22 },
        })
        created.addStyle({ ...selection, atom: "CA" }, { sphere: { color: colour, radius: 0.9 } })
        created.setClickable(selection, true, () => {
          if (live) setSelected(residue)
        })
        created.setHoverable(
          selection,
          true,
          () => created.addLabel(residueLabel(residue), { backgroundOpacity: 0.85 }, selection),
          () => created.removeAllLabels(),
        )
      }

      created.zoomTo({ chain: view.chainId })
      created.render()
    })

    return () => {
      live = false
      viewer?.clear()
    }
  }, [view])

  return { mount, selected }
}

export function StructureViewerPanel({ source }: { source: PreviewSource }) {
  const [kind, setKind] = useState<StructureSource>("rcsb")
  const [pdbId, setPdbId] = useState("")
  const [view, setView] = useState<StructureView | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { mount, selected } = useViewer(view)

  async function load(options: Parameters<typeof fetchStructure>[1]) {
    setBusy(true)
    setError(null)
    try {
      setView(await fetchStructure(source, options))
    } catch (cause) {
      setView(null)
      setError(cause instanceof ApiError ? cause.message : "the structure could not be loaded")
    } finally {
      setBusy(false)
    }
  }

  async function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return
    const format = file.name.toLowerCase().endsWith(".cif") ? "cif" : "pdb"
    await load({ kind: "upload", structureText: await file.text(), structureFormat: format })
  }

  const caveats = view
    ? [
        view.identity < 0.9
          ? `The chain matches the wild type at ${Math.round(view.identity * 100)}% of overlapping residues.`
          : null,
        view.mismatched.length > 0
          ? `${view.mismatched.length} assayed residues differ from the parent sequence.`
          : null,
        view.unmatched.length > 0
          ? `${view.unmatched.length} assayed positions are absent from the model.`
          : null,
      ].filter(Boolean)
    : []

  return (
    <LayerCard className="panel-card chart-card structure-card">
      <div className="panel-heading">
        <div>
          <Text variant="heading3" as="h2">
            Structure
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
            value={kind}
            onValueChange={(value) => setKind(value as StructureSource)}
            tabs={SOURCE_TABS}
          />
          {view && <Badge variant="neutral">Chain {view.chainId}</Badge>}
        </div>
      </div>

      <div className="structure-controls">
        {kind === "rcsb" && (
          <form
            className="structure-fetch"
            onSubmit={(event) => {
              event.preventDefault()
              void load({ kind: "rcsb", pdbId })
            }}
          >
            <Input
              size="sm"
              aria-label="PDB ID"
              placeholder="1PGA"
              value={pdbId}
              maxLength={4}
              onChange={(event) => setPdbId(event.target.value)}
            />
            <Button size="sm" type="submit" disabled={busy || pdbId.length < 4}>
              Fetch
            </Button>
          </form>
        )}
        {kind === "upload" && (
          <label className="structure-upload">
            <input type="file" accept=".pdb,.cif,.ent,.mmcif" onChange={onUpload} />
            <UploadSimple size={16} weight="bold" />
            Choose a PDB or mmCIF file
          </label>
        )}
        {kind === "esmfold" && (
          <div className="structure-fetch">
            <Button size="sm" disabled={busy} onClick={() => void load({ kind: "esmfold" })}>
              Fold wild type
            </Button>
            <Text variant="secondary" size="sm">
              The wild-type sequence is sent to ESM Atlas, a third-party service.
            </Text>
          </div>
        )}
        {view && view.chains.length > 1 && (
          <Select
            size="sm"
            value={view.chainId}
            onValueChange={(value) =>
              value &&
              void load({
                kind: view.method,
                pdbId,
                structureText: view.method === "upload" ? view.pdb : undefined,
                structureFormat: view.format === "mmcif" ? "cif" : "pdb",
                chainId: String(value),
              })
            }
            items={Object.fromEntries(view.chains.map((chain) => [chain, `Chain ${chain}`]))}
          />
        )}
      </div>

      {error && (
        <div className="structure-error">
          <Text variant="secondary" size="sm">
            {error}
          </Text>
        </div>
      )}

      <div className="structure-stage">
        <div className="structure-canvas" ref={mount} />
        {!view && !busy && (
          <div className="structure-placeholder">
            <Text variant="secondary" size="sm">
              {kind === "rcsb"
                ? "Enter a four-character PDB ID."
                : kind === "upload"
                  ? "Choose a coordinate file."
                  : "Fold the wild-type sequence to see it here."}
            </Text>
          </div>
        )}
        {busy && (
          <div className="structure-placeholder">
            <Text variant="secondary" size="sm">
              Loading the structure.
            </Text>
          </div>
        )}
      </div>

      {view && (
        <div className="structure-footer">
          <ul className="structure-legend">
            <li>
              <span className="swatch" style={{ background: DIVERGING[0] }} />
              Deleterious
            </li>
            <li>
              <span className="swatch" style={{ background: DIVERGING[2] }} />
              Neutral
            </li>
            <li>
              <span className="swatch" style={{ background: DIVERGING[4] }} />
              Beneficial
            </li>
            <li>
              <span className="swatch" style={{ background: BACKBONE }} />
              Unassayed
            </li>
          </ul>
          <Text variant="secondary" size="sm">
            {selected
              ? `${selected.parent}${selected.position} at residue ${selected.residue}, ${
                  selected.effect === null
                    ? "no measurement"
                    : `mean effect ${selected.effect.toFixed(2)}`
                }`
              : `${view.label}, ${view.residues.length} of ${
                  view.residues.length + view.unmatched.length
                } assayed sites drawn`}
          </Text>
        </div>
      )}
    </LayerCard>
  )
}

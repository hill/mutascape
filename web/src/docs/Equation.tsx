import { useEffect, useState } from "react"

/** Renders TeX with KaTeX, loaded on demand.
 *
 * The renderer is 300 kB and only the model page needs it, so it stays out of the entry
 * bundle. Its stylesheet is linked from the page, since bundling it inlines every font
 * face as a data URL.
 */
function useRendered(formula: string, displayMode: boolean): string | null {
  const [html, setHtml] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    void import("katex").then((katex) => {
      if (!live) return
      setHtml(
        katex.default.renderToString(formula, {
          displayMode,
          output: "html",
          strict: false,
          throwOnError: false,
        }),
      )
    })
    return () => {
      live = false
    }
  }, [formula, displayMode])

  return html
}

export function Equation({ formula }: { formula: string }) {
  const html = useRendered(formula, true)
  return (
    <div className="docs-equation">
      {html === null ? (
        <span className="docs-equation-source">{formula}</span>
      ) : (
        // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX output built from a literal
        <span dangerouslySetInnerHTML={{ __html: html }} />
      )}
    </div>
  )
}

export function InlineMath({ formula }: { formula: string }) {
  const html = useRendered(formula, false)
  if (html === null) return <code>{formula}</code>
  return (
    // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX output built from a literal
    <span className="docs-inline-math" dangerouslySetInnerHTML={{ __html: html }} />
  )
}

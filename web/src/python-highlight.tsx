import type { ReactNode } from "react"

// Alternation order is the precedence order. Strings and comments come first so a keyword
// inside either is left alone, and the trailing lookahead claims a name only when a call
// follows it.
const pythonTokenPattern =
  /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(#.*$)|\b(from|import|as|def|return|for|in|while|if|elif|else|class|with|None|True|False|and|or|not|is)\b|\b(\d+(?:\.\d+)?)\b|([A-Za-z_]\w*)(?=\s*\()/gm

export function highlightPython(source: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0

  for (const match of source.matchAll(pythonTokenPattern)) {
    const start = match.index
    if (start > cursor) nodes.push(source.slice(cursor, start))
    const kind = match[1]
      ? "string"
      : match[2]
        ? "comment"
        : match[3]
          ? "keyword"
          : match[4]
            ? "number"
            : "function"
    nodes.push(
      <span className={`token-${kind}`} key={`${start}-${kind}`}>
        {match[0]}
      </span>,
    )
    cursor = start + match[0].length
  }

  if (cursor < source.length) nodes.push(source.slice(cursor))
  return nodes
}

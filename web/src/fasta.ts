import { AAS } from "./types";

const CANONICAL = new Set(AAS.split(""));

export interface FastaRecord {
  header: string;
  sequence: string;
  extraRecords: number;
}

export function parseFasta(text: string): FastaRecord {
  const raw = text.trim();
  if (!raw) throw new Error("empty input");

  // If the input doesn't start with a header, treat the whole thing as a bare
  // sequence.
  if (!raw.startsWith(">")) {
    const sequence = validateSequence(raw);
    return { header: "(untitled)", sequence, extraRecords: 0 };
  }

  const chunks = raw.split(/^>/m).filter((c) => c.trim().length > 0);
  const first = chunks[0];
  const nl = first.indexOf("\n");
  const header = nl >= 0 ? first.slice(0, nl).trim() : first.trim();
  const body = nl >= 0 ? first.slice(nl + 1) : "";
  const sequence = validateSequence(body);
  return { header, sequence, extraRecords: Math.max(0, chunks.length - 1) };
}

function validateSequence(body: string): string {
  const cleaned = body.replace(/\s+/g, "").toUpperCase();
  if (cleaned.length < 2) throw new Error("sequence is too short");
  const bad = new Set<string>();
  for (const c of cleaned) if (!CANONICAL.has(c)) bad.add(c);
  if (bad.size > 0) {
    throw new Error(
      `non-canonical amino acids found: ${[...bad].join(", ")}`,
    );
  }
  return cleaned;
}

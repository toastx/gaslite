/* Live Gaslite backend client. The optimized code shown in the diff comes from
   the real server (POST /api/optimize), not the bundled demo dataset. Override
   the host with VITE_GASLITE_API at build time. */

const API_BASE = import.meta.env?.VITE_GASLITE_API?.replace(/\/$/, "") ?? "https://gaslite.fly.dev";

/** Real per-function runtime gas (original vs optimized), from forge's gas report. */
export interface FunctionGas {
  name: string;
  gas_original?: number;
  gas_optimized?: number;
  gas_saved?: number;
}

export interface OptimizeResponse {
  analysis: string;
  suggested_patterns: string[];
  optimized_code: string;
  /** Construction gas of the original — present only when forge verified. */
  gas_before?: number;
  /** Construction gas of the optimized contract — present only when forge verified. */
  gas_after?: number;
  /** Gas saved (positive = improvement) — present only when forge verified. */
  gas_saved?: number;
  /** Per-function runtime gas — present only on an accepted, forge-verified rewrite. */
  per_function_gas?: FunctionGas[];
}

/** Construction-gas figures parsed out of the `analysis` string when present. */
export interface GasFigures {
  before: number;
  after: number;
  saved: number;
}

/** Pull "construction gas 613901 → 346350 (saved 267551)" out of `analysis`. */
export function parseGas(analysis: string): GasFigures | null {
  const m = analysis.match(/gas\s+([\d,]+)\s*(?:→|->)\s*([\d,]+)\s*\(saved\s+([\d,]+)/i);
  // (all three numeric groups use the same comma-tolerant pattern)
  if (!m) return null;
  const n = (s: string) => Number(s.replace(/,/g, ""));
  return { before: n(m[1]), after: n(m[2]), saved: n(m[3]) };
}

export async function optimizeContract(contractSource: string, signal?: AbortSignal): Promise<OptimizeResponse> {
  const source = contractSource.trim();
  if (!source) throw new Error("Contract source is empty — type or paste a contract on the left.");
  const res = await fetch(`${API_BASE}/api/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contract_source: source }),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Gaslite server ${res.status}: ${detail.slice(0, 300) || res.statusText}`);
  }
  return (await res.json()) as OptimizeResponse;
}

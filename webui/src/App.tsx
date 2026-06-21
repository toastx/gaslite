/* Gaslite IDE — top bar, the Monaco diff with idle/analyzing overlays, and the
   savings rail. */
import { useRef, useState, useEffect } from "react";
import { MODEL, TECHNIQUES, REASONS } from "./data";
import { MonacoDiff, type DiffHandle } from "./MonacoDiff";
import { Rail } from "./Rail";
import type { Phase } from "./types";
import { optimizeContract, parseGas, type FunctionGas } from "./api";
import brandLogo from "./gaslite-avatar.png";

function OptimizeBtn({ phase, onOptimize, onReset }: { phase: Phase; onOptimize: () => void; onReset: () => void }) {
  if (phase === "analyzing")
    return (
      <button className="btn btn-primary" disabled>
        <span className="spinner" />
        Analyzing…
      </button>
    );
  if (phase === "done")
    return (
      <button className="btn btn-ghost" onClick={onReset}>
        ↺ Reset
      </button>
    );
  return (
    <button className="btn btn-primary" onClick={onOptimize}>
      Optimize contract
    </button>
  );
}

export function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [runCount, setRunCount] = useState(100000);
  const [optimized, setOptimized] = useState<string | undefined>();
  const [analysis, setAnalysis] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [mntUsd, setMntUsd] = useState(MODEL.mntUsd);
  const [gasPriceGwei, setGasPriceGwei] = useState(MODEL.gasPriceGwei);
  const [gasBefore, setGasBefore] = useState<number | undefined>();
  const [gasAfter, setGasAfter] = useState<number | undefined>();
  const [gasSaved, setGasSaved] = useState<number | undefined>();
  const [patterns, setPatterns] = useState<string[]>([]);
  const [fnGas, setFnGas] = useState<FunctionGas[]>([]);
  const diffRef = useRef<DiffHandle>(null);
  const done = phase === "done";

  useEffect(() => {
    fetch("https://api.coingecko.com/api/v3/simple/price?ids=mantle&vs_currencies=usd")
      .then((r) => r.json())
      .then((d) => {
        const price = d?.mantle?.usd;
        if (typeof price === "number" && price > 0) setMntUsd(price);
      })
      .catch(() => {/* keep default */});
  }, []);

  // Live Mantle gas price via the public RPC (eth_gasPrice returns wei → Gwei).
  useEffect(() => {
    const rpc = import.meta.env?.VITE_MANTLE_RPC ?? "https://rpc.mantle.xyz";
    fetch(rpc, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_gasPrice", params: [] }),
    })
      .then((r) => r.json())
      .then((d) => {
        const gwei = parseInt(d?.result, 16) / 1e9;
        if (Number.isFinite(gwei) && gwei > 0) setGasPriceGwei(gwei);
      })
      .catch(() => {/* keep default */});
  }, []);

  const optimize = async () => {
    setError(null);
    setPhase("analyzing");
    const source = diffRef.current?.getOriginal() ?? "";
    try {
      const res = await optimizeContract(source);
      setOptimized(res.optimized_code);
      setAnalysis(res.analysis);
      setGasBefore(res.gas_before);
      setGasAfter(res.gas_after);
      setGasSaved(res.gas_saved);
      setPatterns(res.suggested_patterns);
      setFnGas(res.per_function_gas ?? []);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  };
  const reset = () => {
    setPhase("idle");
    setOptimized(undefined);
    setAnalysis("");
    setGasBefore(undefined);
    setGasAfter(undefined);
    setGasSaved(undefined);
    setPatterns([]);
    setFnGas([]);
    setError(null);
  };

  return (
    <>
      {/* topbar */}
      <div className="topbar">
        <div className="brand">
          <span className="gl-mark">
            <img src={brandLogo} alt="" className="gl-dot" />
            Gaslite
          </span>
        </div>
        <div className="top-right">
          <span className="net">
            <span className="dotg" />
            Mantle Mainnet
          </span>
          <a
            className="analytics-link"
            href="https://gaslite-analytics.fly.dev"
            target="_blank"
            rel="noopener noreferrer"
          >
            Analytics
          </a>
          <OptimizeBtn phase={phase} onOptimize={optimize} onReset={reset} />
        </div>
      </div>

      {/* main */}
      <div className="main">
        {/* editors (two columns, shared scroll) */}
        <div className="editors" style={{ gridColumn: "1 / span 2", display: "flex", flexDirection: "column" }}>
          <div className="ed-headrow">
            <div className="ed-head">
              <span>Original</span>
              <span className="mono" style={{ color: "var(--ink-3)" }}>
                baseline
              </span>
            </div>
            <div className="ed-head">
              <span style={{ color: done ? "var(--accent-ink)" : "var(--ink-2)" }}>Optimized by Gaslite</span>
              {done &&
                (() => {
                  const g = parseGas(analysis);
                  const realGas = g && g.before > 0;
                  const pct = realGas
                    ? Math.round((g.saved / g.before) * 100)
                    : Math.round(MODEL.savedPct(runCount) * 100);
                  return (
                    <span className="pill-save" title={analysis || undefined}>
                      −{pct}% {realGas ? "deploy gas" : ""}
                    </span>
                  );
                })()}
            </div>
          </div>
          <div className="diff-shell" style={{ position: "relative", flex: 1, overflow: "hidden" }}>
            <MonacoDiff ref={diffRef} phase={phase} optimizedSrc={optimized} />
            {phase === "analyzing" && <div className="scan" />}
            {phase !== "done" && (
              <div className="ov" style={{ left: "50%" }}>
                {phase === "analyzing" ? (
                  <>
                    <div className="ring" />
                    <h3>Analyzing bytecode…</h3>
                    <p>Gaslite is rewriting storage layout, calldata usage and control flow.</p>
                    <div className="tech-chips">
                      {TECHNIQUES.map((t, i) => (
                        <span key={t} className="tech-chip" style={{ animationDelay: i * 140 + "ms" }}>
                          {REASONS[t].tag}
                        </span>
                      ))}
                    </div>
                  </>
                ) : error ? (
                  <>
                    <h3>Couldn’t reach Gaslite</h3>
                    <p style={{ color: "var(--danger, #c5221f)" }}>{error}</p>
                    <p>Edit the contract on the left and try again.</p>
                  </>
                ) : (
                  <>
                    <h3>Optimized output</h3>
                    <p>
                      Press <b>Optimize contract</b> to rewrite this Solidity for minimum gas — behaviour stays
                      identical.
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* stats rail */}
        {done ? (
          <Rail
            done={done}
            runCount={runCount}
            setRunCount={setRunCount}
            mntUsd={mntUsd}
            gasPriceGwei={gasPriceGwei}
            gasBefore={gasBefore}
            gasAfter={gasAfter}
            gasSaved={gasSaved}
            patterns={patterns}
            fnGas={fnGas}
            analysis={analysis}
          />
        ) : (
          <div className="rail">
            <div className="rail-empty">
              <div className="glyph" />
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)" }}>Savings appear here</div>
              <div style={{ fontSize: 12, maxWidth: 220 }}>
                Run Gaslite to see gas saved per call, deploy savings and a live cost simulation.
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

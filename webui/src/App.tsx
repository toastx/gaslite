/* Gaslite IDE — top bar, the Monaco diff with idle/analyzing overlays, and the
   savings rail. */
import { useState } from "react";
import { MODEL, TECHNIQUES, REASONS } from "./data";
import { MonacoDiff } from "./MonacoDiff";
import { Rail } from "./Rail";
import type { Phase } from "./types";

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
  const [fnIdx, setFnIdx] = useState(0);
  const done = phase === "done";

  const optimize = () => {
    setPhase("analyzing");
    setTimeout(() => setPhase("done"), 1600);
  };
  const reset = () => setPhase("idle");

  return (
    <>
      {/* topbar */}
      <div className="topbar">
        <div className="brand">
          <span className="gl-mark">
            <span className="gl-dot" />
            Gaslite
          </span>
        </div>
        <div className="top-right">
          <span className="net">
            <span className="dotg" />
            Mantle Mainnet
          </span>
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
              {done && <span className="pill-save">−{Math.round(MODEL.savedPct(runCount) * 100)}%</span>}
            </div>
          </div>
          <div className="diff-shell" style={{ position: "relative", flex: 1, overflow: "hidden" }}>
            <MonacoDiff phase={phase} />
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
          <Rail done={done} runCount={runCount} setRunCount={setRunCount} fnIdx={fnIdx} setFnIdx={setFnIdx} />
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

/* Gaslite — savings rail: headline summary, deploy meter, patterns applied,
   estimated cost saved and a live gas/cost simulation. */
import { MODEL, PRESETS } from "./data";
import { AnimatedNum, BarMeter, SavingsGraph, fmt, fmtGas, fmtMnt } from "./lib";
import logo from "./gaslite-mark-white.png";

interface RailProps {
  done: boolean;
  runCount: number;
  setRunCount: (n: number) => void;
  mntUsd: number;
  gasBefore?: number;
  gasAfter?: number;
  gasSaved?: number;
  patterns?: string[];
  analysis?: string;
}

/** Convert a knowledge-base pattern slug to a readable label.
 *  e.g. "erc721-calldata-array-001" → "calldata array" */
function patternLabel(id: string): string {
  return id
    .replace(/^(erc\d+|evm|mantle|general)-/i, "")
    .replace(/-\d+$/, "")
    .replace(/-/g, " ");
}

export function Rail({
  done,
  runCount,
  setRunCount,
  mntUsd,
  gasBefore,
  gasAfter,
  gasSaved,
  patterns = [],
  analysis = "",
}: RailProps) {
  // Use real forge-measured deploy gas when available, fall back to demo model.
  const realDeploy = gasBefore != null && gasAfter != null;
  const deployBefore = realDeploy ? gasBefore! : MODEL.deploy.before;
  const deployAfter = realDeploy ? gasAfter! : MODEL.deploy.after;
  const depPct = (1 - deployAfter / deployBefore) * 100;

  // Hero % — real deploy savings ratio when we have forge numbers.
  const savedPct = realDeploy ? depPct : MODEL.savedPct(runCount) * 100;

  // Cost estimation: prefer real deploy gas saved; fall back to blended model.
  const realGasSaved = gasSaved != null && gasSaved > 0 ? gasSaved : null;
  const costGasSaved = realGasSaved ?? MODEL.savedGas(runCount);
  const mnt = MODEL.gasToMnt(costGasSaved);
  const usd = mnt * mntUsd;

  const sliderVal = runCount <= 1 ? 0 : Math.round((Math.log10(runCount) / 6) * 600);
  const onSlide = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = +e.target.value;
    setRunCount(v === 0 ? 1 : Math.round(Math.pow(10, (v / 600) * 6)));
  };
  const runLabel = runCount === 0 ? "deploy only" : fmt(runCount) + " calls";

  const hasPatterns = patterns.length > 0;

  return (
    <div className="rail">
      <div className="kicker">Savings summary</div>

      {/* hero logo */}
      <div className="card hero card-anim" style={{ animationDelay: "0ms" }}>
        <img src={logo} alt="Gaslite" className="hero-logo" />
        <div className="meta">
          <div className="big">Up to {Math.round(savedPct)}% cheaper</div>
          <div className="sub">
            {realDeploy
              ? `Forge-verified on Mantle fork. ${patterns.length > 0 ? `${patterns.length} pattern(s) applied.` : ""}`
              : `Blended across deployment and ${runLabel}. 7 optimizations applied automatically.`}
          </div>
        </div>
      </div>

      {/* deploy + patterns */}
      <div className="row2">
        <div className="card card-anim" style={{ animationDelay: "70ms" }}>
          <div className="lab">Deployment</div>
          <div className="val pos">
            −<AnimatedNum value={depPct} run={done} format={(x) => x.toFixed(0)} />%
          </div>
          <div className="delta">
            {fmtGas(deployBefore)} → {fmtGas(deployAfter)}
          </div>
          <div style={{ marginTop: 10 }}>
            <BarMeter before={deployBefore} after={deployAfter} run={done} />
          </div>
        </div>

        {/* Patterns applied — replaces the static per-function selector */}
        <div className="card card-anim" style={{ animationDelay: "130ms" }}>
          <div className="lab">Patterns applied</div>
          {hasPatterns ? (
            <>
              <div className="val pos" style={{ fontSize: 28 }}>
                <AnimatedNum value={patterns.length} run={done} format={(x) => Math.round(x).toString()} duration={500} />
              </div>
              <div className="delta">gas optimization techniques</div>
              <div className="fnsel" style={{ marginTop: 10, flexWrap: "wrap", gap: 4 }}>
                {patterns.map((p) => (
                  <span key={p} className="fnbtn on" title={p} style={{ cursor: "default", textTransform: "capitalize" }}>
                    {patternLabel(p)}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="val pos" style={{ fontSize: 28 }}>7</div>
              <div className="delta">gas optimization techniques</div>
            </>
          )}
        </div>
      </div>

      {/* cost saved */}
      <div className="card card-anim" style={{ animationDelay: "190ms" }}>
        <div className="lab">
          {realGasSaved ? "Estimated cost saved · deploy" : `Estimated cost saved · ${runLabel}`}
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginTop: 6 }}>
          <div className="val mono" style={{ color: "var(--ink)" }}>
            <AnimatedNum value={mnt} run={done} format={fmtMnt} />{" "}
            <span style={{ fontSize: 13, color: "var(--ink-3)" }}>MNT</span>
          </div>
          <div className="val mono" style={{ fontSize: 18, color: "var(--ink-3)" }}>
            ≈ $<AnimatedNum value={usd} run={done} format={(x) => (x < 0.01 ? x.toFixed(4) : x.toFixed(2))} />
          </div>
        </div>
        <div className="delta">
          <AnimatedNum value={costGasSaved} run={done} format={fmtGas} /> gas saved
          {realGasSaved ? " (forge-measured)" : " (estimated)"}
        </div>
        {analysis && (
          <div className="assume" style={{ marginTop: 8, lineHeight: 1.5 }}>
            {analysis}
          </div>
        )}
      </div>

      {/* simulation */}
      <div className="card card-anim" style={{ animationDelay: "250ms" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
          <div className="kicker" style={{ color: "var(--ink-2)" }}>
            Simulation
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--accent-ink)", fontWeight: 600 }}>
            {runLabel}
          </div>
        </div>
        <SavingsGraph runCount={Math.max(runCount, 1)} run={done} width={350} height={172} />
        <div className="legend">
          <span>
            <i style={{ background: "var(--del-bar)" }} />
            Original
          </span>
          <span>
            <i style={{ background: "var(--accent)" }} />
            Gaslite
          </span>
        </div>
        <input
          className="slider"
          type="range"
          min="0"
          max="600"
          step="1"
          value={sliderVal}
          onChange={onSlide}
          style={{ marginTop: 12 }}
        />
        <div className="presets">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              className={"preset" + (runCount === p.runs || (p.runs === 0 && runCount <= 1) ? " on" : "")}
              onClick={() => setRunCount(p.runs)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="simrow">
          <span className="k">Original</span>
          <span className="v" style={{ color: "var(--del-ink)" }}>
            {fmt(MODEL.cumBefore(runCount))} gas
          </span>
        </div>
        <div className="simrow">
          <span className="k">With Gaslite</span>
          <span className="v pos">{fmt(MODEL.cumAfter(runCount))} gas</span>
        </div>
        <div className="assume">
          assumes {MODEL.gasPriceGwei} Gwei · MNT ${mntUsd.toFixed(4)} on Mantle
        </div>
      </div>
    </div>
  );
}

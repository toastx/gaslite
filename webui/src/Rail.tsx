/* Gaslite — savings rail: headline donut, deploy/per-call meters, estimated
   cost saved and a live gas/cost simulation. */
import { MODEL, FUNCS, PRESETS } from "./data";
import { AnimatedNum, BarMeter, SavingsGraph, fmt, fmtGas, fmtMnt } from "./lib";
import logo from "./gaslite-mark-white.png";

interface RailProps {
  done: boolean;
  runCount: number;
  setRunCount: (n: number) => void;
  fnIdx: number;
  setFnIdx: (i: number) => void;
}

export function Rail({ done, runCount, setRunCount, fnIdx, setFnIdx }: RailProps) {
  const savedGas = MODEL.savedGas(runCount);
  const savedPct = MODEL.savedPct(runCount) * 100;
  const mnt = MODEL.gasToMnt(savedGas),
    usd = MODEL.gasToUsd(savedGas);
  const fn = FUNCS[fnIdx]!,
    fnPct = (1 - fn.after / fn.before) * 100;
  const dep = MODEL.deploy,
    depPct = (1 - dep.after / dep.before) * 100;

  // slider position from runCount (log)
  const sliderVal = runCount <= 1 ? 0 : Math.round((Math.log10(runCount) / 6) * 600);
  const onSlide = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = +e.target.value;
    setRunCount(v === 0 ? 1 : Math.round(Math.pow(10, (v / 600) * 6)));
  };
  const runLabel = runCount === 0 ? "deploy only" : fmt(runCount) + " calls";

  return (
    <div className="rail">
      <div className="kicker">Savings summary</div>

      {/* hero logo */}
      <div className="card hero card-anim" style={{ animationDelay: "0ms" }}>
        <img src={logo} alt="Gaslite" className="hero-logo" />
        <div className="meta">
          <div className="big">Up to {Math.round(savedPct)}% cheaper</div>
          <div className="sub">
            Blended across deployment and {runLabel}. 7 optimizations applied automatically.
          </div>
        </div>
      </div>

      {/* deploy + per-call */}
      <div className="row2">
        <div className="card card-anim" style={{ animationDelay: "70ms" }}>
          <div className="lab">Deployment</div>
          <div className="val pos">
            −<AnimatedNum value={depPct} run={done} format={(x) => x.toFixed(0)} />%
          </div>
          <div className="delta">
            {fmtGas(dep.before)} → {fmtGas(dep.after)}
          </div>
          <div style={{ marginTop: 10 }}>
            <BarMeter before={dep.before} after={dep.after} run={done} />
          </div>
        </div>
        <div className="card card-anim" style={{ animationDelay: "130ms" }}>
          <div className="lab">Per call</div>
          <div className="val pos">
            −<AnimatedNum value={fnPct} run={done} format={(x) => x.toFixed(0)} duration={500} />%
          </div>
          <div className="delta">{fmt(fn.before - fn.after)} gas saved</div>
          <div style={{ marginTop: 10 }}>
            <BarMeter before={fn.before} after={fn.after} run={done} />
          </div>
          <div className="fnsel">
            {FUNCS.map((f, i) => (
              <button key={f.name} className={"fnbtn" + (i === fnIdx ? " on" : "")} onClick={() => setFnIdx(i)}>
                {f.name}()
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* cost saved */}
      <div className="card card-anim" style={{ animationDelay: "190ms" }}>
        <div className="lab">Estimated cost saved · {runLabel}</div>
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
          <AnimatedNum value={savedGas} run={done} format={fmtGas} /> gas saved total
        </div>
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
          assumes {MODEL.gasPriceGwei} Gwei · MNT ${MODEL.mntUsd} on Mantle
        </div>
      </div>
    </div>
  );
}

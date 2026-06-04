/* Gaslite — UI primitives: formatting helpers, a count-up hook, animated
   number, donut, bar meter and the log-log savings graph. */
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { MODEL } from "./data";

/* ---------- formatting ---------- */
export function fmt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
export function fmtGas(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return Math.round(n).toString();
}
export function fmtMnt(n: number): string {
  if (n >= 1) return n.toFixed(2);
  if (n >= 0.0001) return n.toFixed(4);
  if (n > 0) return "<0.0001";
  return "0";
}

/* ---------- count-up hook (easeOutExpo) ---------- */
interface CountUpOpts {
  duration?: number;
  run?: boolean;
}
export function useCountUp(target: number, { duration = 900, run = true }: CountUpOpts = {}): number {
  const staticMode =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [val, setVal] = useState(run && !staticMode ? 0 : target);
  const fromRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!run || staticMode) {
      setVal(target);
      fromRef.current = target;
      return;
    }
    const from = fromRef.current;
    const t0 = performance.now();
    cancelAnimationFrame(rafRef.current);
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(2, -10 * p);
      const v = from + (target - from) * (p >= 1 ? 1 : e);
      setVal(v);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, run]);

  // keep fromRef synced so a re-tween starts from the current display value
  useEffect(() => {
    fromRef.current = val;
  });
  return val;
}

/* ---------- animated number ---------- */
interface AnimatedNumProps {
  value: number;
  run?: boolean;
  format?: (n: number) => string | number;
  duration?: number;
  className?: string;
  style?: CSSProperties;
}
export function AnimatedNum({ value, run, format = fmt, duration = 900, className, style }: AnimatedNumProps) {
  const v = useCountUp(value, { run, duration });
  return (
    <span className={className} style={style}>
      {format(v)}
    </span>
  );
}

/* ---------- donut / arc for headline % ---------- */
export function Donut({ pct, run, size = 132, stroke = 11 }: { pct: number; run?: boolean; size?: number; stroke?: number }) {
  const v = useCountUp(pct, { run, duration: 1100 });
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const off = c * (1 - v / 100);
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,.22)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#fff"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={off}
          style={{ transition: "none" }}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
        }}
      >
        <div style={{ fontSize: 34, fontWeight: 800, letterSpacing: "-.03em", lineHeight: 1 }}>
          {v.toFixed(0)}
          <span style={{ fontSize: 18 }}>%</span>
        </div>
        <div style={{ fontSize: 10.5, fontWeight: 600, opacity: 0.82, marginTop: 3, letterSpacing: ".04em" }}>
          LESS&nbsp;GAS
        </div>
      </div>
    </div>
  );
}

/* ---------- horizontal bar meter (before vs after) ---------- */
export function BarMeter({ before, after, run }: { before: number; after: number; run?: boolean }) {
  const pct = after / before; // optimized width as fraction of original
  const w = useCountUp(pct * 100, { run, duration: 850 });
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      <div style={{ height: 9, borderRadius: 9, background: "var(--del-bg)", position: "relative", overflow: "hidden" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "repeating-linear-gradient(90deg,transparent 0 7px,rgba(0,0,0,.03) 7px 8px)",
          }}
        />
      </div>
      <div style={{ height: 9, borderRadius: 9, background: "var(--surf-2)", position: "relative", overflow: "hidden" }}>
        <div
          style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: w + "%", background: "var(--accent)", borderRadius: 9 }}
        />
      </div>
    </div>
  );
}

/* ---------- log-log savings graph ---------- */
export function SavingsGraph({ runCount, run, width = 300, height = 168 }: { runCount: number; run?: boolean; width?: number; height?: number }) {
  const padL = 46,
    padR = 12,
    padT = 14,
    padB = 24;
  const plotW = width - padL - padR,
    plotH = height - padT - padB;
  const logMax = 6; // log10(1e6)
  const yMin = 6,
    yMax = 11; // gas magnitude 1e6 .. 1e11
  const xOf = (n: number) => padL + (Math.log10(n + 1) / logMax) * plotW;
  const yOf = (g: number) => padT + (1 - (Math.log10(Math.max(g, 1)) - yMin) / (yMax - yMin)) * plotH;

  const pts = useMemo(() => {
    const a: string[] = [],
      b: string[] = [];
    for (let i = 0; i <= 48; i++) {
      const n = Math.pow(10, (i / 48) * logMax) - 1;
      a.push(`${xOf(n).toFixed(1)},${yOf(MODEL.cumBefore(n)).toFixed(1)}`);
      b.push(`${xOf(n).toFixed(1)},${yOf(MODEL.cumAfter(n)).toFixed(1)}`);
    }
    return { before: a.join(" "), after: b.join(" ") };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height]);

  const mx = xOf(runCount);
  const myB = yOf(MODEL.cumBefore(runCount));
  const myA = yOf(MODEL.cumAfter(runCount));
  const xticks = [1, 10, 100, 1000, 10000, 100000, 1000000];
  const xlab = ["1", "10", "100", "1K", "10K", "100K", "1M"];
  const yticks = [6, 7, 8, 9, 10, 11];
  const ylab = ["1M", "10M", "100M", "1B", "10B", "100B"];

  const tr = { transition: "all .35s cubic-bezier(.2,.7,.3,1)" } as const;
  return (
    <svg width={width} height={height} className={run ? "graph-draw" : ""} style={{ display: "block" }}>
      {yticks.map((t, i) => {
        const y = padT + (1 - (t - yMin) / (yMax - yMin)) * plotH;
        return (
          <g key={t}>
            <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="var(--line)" strokeWidth="1" />
            <text x={padL - 8} y={y + 3} fontSize="9" fill="var(--ink-3)" textAnchor="end" fontFamily="var(--mono)">
              {ylab[i]}
            </text>
          </g>
        );
      })}
      {xticks.map((t, i) => (
        <text key={t} x={xOf(t)} y={height - 6} fontSize="9" fill="var(--ink-3)" textAnchor="middle" fontFamily="var(--mono)">
          {xlab[i]}
        </text>
      ))}
      <polyline className="gline gline-before" points={pts.before} fill="none" stroke="var(--del-bar)" strokeWidth="2" />
      <polyline className="gline gline-after" points={pts.after} fill="none" stroke="var(--accent)" strokeWidth="2.6" />
      <line x1={mx} y1={myB} x2={mx} y2={myA} stroke="var(--add-ink)" strokeWidth="1.5" strokeDasharray="3 3" style={tr} />
      <circle cx={mx} cy={myB} r="3.2" fill="#fff" stroke="var(--del-bar)" strokeWidth="2" style={tr} />
      <circle cx={mx} cy={myA} r="3.6" fill="var(--accent)" stroke="#fff" strokeWidth="1.6" style={tr} />
    </svg>
  );
}

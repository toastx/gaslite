/* Gaslite — UI primitives: Solidity highlighter, count-up hook, animated
   number, donut, bar meter, log-log savings graph, and the side-by-side
   diff table with hover-to-explain. Exported to window. */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* ---------- formatting ---------- */
function fmt(n) { return Math.round(n).toLocaleString("en-US"); }
function fmtGas(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return Math.round(n).toString();
}
function fmtMnt(n) {
  if (n >= 1) return n.toFixed(2);
  if (n >= 0.0001) return n.toFixed(4);
  if (n > 0) return "<0.0001";
  return "0";
}

/* ---------- Solidity syntax highlighter ---------- */
const SOL_RX = /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*")|\b(function|public|external|internal|private|view|pure|returns|return|memory|calldata|storage|for|if|else|revert|require|unchecked|contract|struct|mapping|emit|immutable|constant|while|new|msg)\b|\b(address|bool|string|bytes\d*|uint\d*|int\d*)\b|\b(\d[\d_]*)\b|\b([A-Za-z_]\w*)(?=\s*\()/g;
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function hl(code) {
  if (!code) return "&nbsp;";
  return esc(code).replace(SOL_RX, (m, cm, st, kw, ty, nu, fn) => {
    if (cm) return `<span class="t-cm">${cm}</span>`;
    if (st) return `<span class="t-st">${st}</span>`;
    if (kw) return `<span class="t-kw">${kw}</span>`;
    if (ty) return `<span class="t-ty">${ty}</span>`;
    if (nu) return `<span class="t-nu">${nu}</span>`;
    if (fn) return `<span class="t-fn">${fn}</span>`;
    return m;
  });
}

/* ---------- count-up hook (easeOutExpo) ---------- */
function useCountUp(target, { duration = 900, run = true, deps = [] } = {}) {
  const staticMode = typeof window !== "undefined" && (window.__glStatic ||
    (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches));
  const [val, setVal] = useState(run && !staticMode ? 0 : target);
  const fromRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    if (!run || staticMode) { setVal(target); fromRef.current = target; return; }
    const from = fromRef.current;
    const t0 = performance.now();
    cancelAnimationFrame(rafRef.current);
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(2, -10 * p);
      const v = from + (target - from) * (p >= 1 ? 1 : e);
      setVal(v);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line
  }, [target, run, ...deps]);
  // keep fromRef synced so slider re-tweens from current display value
  useEffect(() => { fromRef.current = val; });
  return val;
}

/* ---------- animated number ---------- */
function AnimatedNum({ value, run, format = fmt, duration = 900, className, style }) {
  const v = useCountUp(value, { run, duration });
  return <span className={className} style={style}>{format(v)}</span>;
}

/* ---------- donut / arc for headline % ---------- */
function Donut({ pct, run, size = 132, stroke = 11 }) {
  const v = useCountUp(pct, { run, duration: 1100 });
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const off = c * (1 - v / 100);
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,.22)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#fff" strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off}
          style={{ transition: "none" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", color: "#fff" }}>
        <div style={{ fontSize: 34, fontWeight: 800, letterSpacing: "-.03em", lineHeight: 1 }}>
          {v.toFixed(0)}<span style={{ fontSize: 18 }}>%</span></div>
        <div style={{ fontSize: 10.5, fontWeight: 600, opacity: .82, marginTop: 3, letterSpacing: ".04em" }}>LESS&nbsp;GAS</div>
      </div>
    </div>
  );
}

/* ---------- horizontal bar meter (before vs after) ---------- */
function BarMeter({ before, after, run }) {
  const pct = after / before; // optimized width as fraction of original
  const w = useCountUp(run ? pct * 100 : pct * 100, { run, duration: 850 });
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      <div style={{ height: 9, borderRadius: 9, background: "var(--del-bg)", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", inset: 0, background: "repeating-linear-gradient(90deg,transparent 0 7px,rgba(0,0,0,.03) 7px 8px)" }} />
      </div>
      <div style={{ height: 9, borderRadius: 9, background: "var(--surf-2)", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: w + "%",
          background: "var(--accent)", borderRadius: 9 }} />
      </div>
    </div>
  );
}

/* ---------- log-log savings graph ---------- */
function SavingsGraph({ runCount, run, width = 300, height = 168 }) {
  const M = window.GL.MODEL;
  const padL = 46, padR = 12, padT = 14, padB = 24;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const logMax = 6; // log10(1e6)
  const yMin = 6, yMax = 11; // gas magnitude 1e6 .. 1e11
  const xOf = (n) => padL + (Math.log10(n + 1) / logMax) * plotW;
  const yOf = (g) => padT + (1 - (Math.log10(Math.max(g, 1)) - yMin) / (yMax - yMin)) * plotH;
  const pts = useMemo(() => {
    const a = [], b = [];
    for (let i = 0; i <= 48; i++) {
      const n = Math.pow(10, (i / 48) * logMax) - 1;
      a.push(`${xOf(n).toFixed(1)},${yOf(M.cumBefore(n)).toFixed(1)}`);
      b.push(`${xOf(n).toFixed(1)},${yOf(M.cumAfter(n)).toFixed(1)}`);
    }
    return { before: a.join(" "), after: b.join(" ") };
  }, [width, height]);
  const mx = xOf(runCount);
  const myB = yOf(M.cumBefore(runCount));
  const myA = yOf(M.cumAfter(runCount));
  const xticks = [1, 10, 100, 1000, 10000, 100000, 1000000];
  const xlab = ["1", "10", "100", "1K", "10K", "100K", "1M"];
  const yticks = [6, 7, 8, 9, 10, 11];
  const ylab = ["1M", "10M", "100M", "1B", "10B", "100B"];
  return (
    <svg width={width} height={height} className={run ? "graph-draw" : ""} style={{ display: "block" }}>
      {/* y grid */}
      {yticks.map((t, i) => {
        const y = padT + (1 - (t - yMin) / (yMax - yMin)) * plotH;
        return <g key={t}>
          <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="var(--line)" strokeWidth="1" />
          <text x={padL - 8} y={y + 3} fontSize="9" fill="var(--ink-3)" textAnchor="end" fontFamily="var(--mono)">{ylab[i]}</text>
        </g>;
      })}
      {/* x labels */}
      {xticks.map((t, i) => (
        <text key={t} x={xOf(t)} y={height - 6} fontSize="9" fill="var(--ink-3)" textAnchor="middle" fontFamily="var(--mono)">{xlab[i]}</text>
      ))}
      {/* before line */}
      <polyline className="gline gline-before" points={pts.before} fill="none" stroke="var(--del-bar)" strokeWidth="2" />
      {/* after line */}
      <polyline className="gline gline-after" points={pts.after} fill="none" stroke="var(--accent)" strokeWidth="2.6" />
      {/* marker */}
      <line x1={mx} y1={myB} x2={mx} y2={myA} stroke="var(--add-ink)" strokeWidth="1.5" strokeDasharray="3 3"
        style={{ transition: "all .35s cubic-bezier(.2,.7,.3,1)" }} />
      <circle cx={mx} cy={myB} r="3.2" fill="#fff" stroke="var(--del-bar)" strokeWidth="2" style={{ transition: "all .35s cubic-bezier(.2,.7,.3,1)" }} />
      <circle cx={mx} cy={myA} r="3.6" fill="var(--accent)" stroke="#fff" strokeWidth="1.6" style={{ transition: "all .35s cubic-bezier(.2,.7,.3,1)" }} />
    </svg>
  );
}

/* ---------- diff table (side-by-side, hover-to-explain) ---------- */
function DiffRow({ row, ln, rn, revealed, idx, onWhy }) {
  const [hover, setHover] = useState(false);
  if (row.kind === "gap") {
    return (
      <div className="diff-gap">
        <span className="diff-gap-line" /><span>{row.l}</span><span className="diff-gap-line" />
      </div>
    );
  }
  const why = row.why ? window.GL.REASONS[row.why] : null;
  const leftType = revealed && (row.kind === "del" || row.kind === "chg") ? "del" : "ctx";
  const rightType = row.kind === "add" || row.kind === "chg" ? "add" : "ctx";
  const interactive = !!why && revealed;
  const delay = revealed ? Math.min(idx * 45, 900) : 0;
  return (
    <div
      className={"diff-row" + (interactive ? " has-why" : "") + (hover ? " row-hover" : "")}
      onMouseEnter={() => { if (interactive) { setHover(true); onWhy && onWhy(row.why); } }}
      onMouseLeave={() => { setHover(false); onWhy && onWhy(null); }}
    >
      {/* left / original */}
      <div className={"cell cell-" + leftType}>
        <span className="gut">{row.l === null ? "" : ln}</span>
        <code className="src" dangerouslySetInnerHTML={{ __html: row.l === null ? "&nbsp;" : hl(row.l) }} />
      </div>
      {/* right / optimized */}
      <div className={"cell cell-" + (revealed ? rightType : "pending")}
        style={revealed ? { animationDelay: delay + "ms" } : undefined}>
        <span className="gut">{!revealed ? "" : (row.r === null ? "" : rn)}</span>
        <code className="src"
          dangerouslySetInnerHTML={{ __html: !revealed ? "&nbsp;" : (row.r === null ? "&nbsp;" : hl(row.r)) }} />
        {interactive && hover && why && (
          <div className="why-pop">
            <div className="why-tag">{why.tag}</div>
            <div className="why-title">{why.title}</div>
            <div className="why-body">{why.body}</div>
            <div className="why-gas"><span>Saves</span><b>{why.gas}</b></div>
          </div>
        )}
      </div>
    </div>
  );
}

function DiffTable({ revealed, onWhy }) {
  const rows = window.GL.ROWS;
  let l = 1, r = 1;
  const out = [];
  rows.forEach((row, i) => {
    let ln = "", rn = "";
    if (row.kind === "gap") { /* keep numbers */ }
    else {
      if (row.l !== null) { ln = l; l++; }
      if (row.r !== null) { rn = r; r++; }
    }
    out.push(<DiffRow key={i} row={row} ln={ln} rn={rn} revealed={revealed} idx={i} onWhy={onWhy} />);
  });
  return <div className="diff-table">{out}</div>;
}

Object.assign(window, {
  glFmt: fmt, fmtGas, fmtMnt, hl,
  useCountUp, AnimatedNum, Donut, BarMeter, SavingsGraph, DiffTable,
});

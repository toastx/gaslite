/* Gaslite — Monaco-backed diff editor.
   A real side-by-side diff editor with an editable original pane, a custom
   "gaslite" light theme matching the app palette, a minimal Solidity Monarch
   grammar, and the hover-to-explain optimizations rebuilt as line-anchored
   decorations + a custom popover. */
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { loadMonaco } from "./monaco";
import { ORIGINAL_SRC, OPTIMIZATIONS, REASONS, type ReasonKey } from "./data";
import type { Phase } from "./types";

/** Imperative handle so the app can read the (user-edited) original source. */
export interface DiffHandle {
  getOriginal: () => string;
}

/* ---- one-time language + theme registration ---- */
let glMonacoReady = false;
function glSetupMonaco(monaco: any) {
  if (glMonacoReady) return;
  glMonacoReady = true;

  monaco.languages.register({ id: "sol" });
  monaco.languages.setLanguageConfiguration("sol", {
    comments: { lineComment: "//", blockComment: ["/*", "*/"] },
    brackets: [
      ["{", "}"],
      ["[", "]"],
      ["(", ")"],
    ],
    autoClosingPairs: [
      { open: "{", close: "}" },
      { open: "[", close: "]" },
      { open: "(", close: ")" },
      { open: '"', close: '"' },
    ],
  });
  monaco.languages.setMonarchTokensProvider("sol", {
    keywords: [
      "pragma", "solidity", "contract", "struct", "mapping", "function", "constructor",
      "modifier", "event", "error", "emit", "return", "returns", "public", "external",
      "internal", "private", "view", "pure", "payable", "memory", "calldata", "storage",
      "immutable", "constant", "for", "while", "if", "else", "require", "revert",
      "unchecked", "new", "indexed", "msg", "block",
    ],
    typeKeywords: ["address", "bool", "string", "bytes", "uint", "int", "byte", "fixed", "ufixed"],
    tokenizer: {
      root: [
        [/\/\/.*$/, "comment"],
        [/\/\*/, "comment", "@comment"],
        [/"(?:[^"\\]|\\.)*"/, "string"],
        [/'(?:[^'\\]|\\.)*'/, "string"],
        [/\b(?:uint|int|bytes|fixed|ufixed)\d*\b/, "type"],
        [/[a-zA-Z_]\w*(?=\s*\()/, { cases: { "@keywords": "keyword", "@typeKeywords": "type", "@default": "function" } }],
        [/[a-zA-Z_]\w*/, { cases: { "@keywords": "keyword", "@typeKeywords": "type", "@default": "identifier" } }],
        [/\d[\d_]*/, "number"],
        [/[{}()\[\]]/, "delimiter"],
      ],
      comment: [
        [/[^*]+/, "comment"],
        [/\*\//, "comment", "@pop"],
        [/./, "comment"],
      ],
    },
  });

  monaco.editor.defineTheme("gaslite", {
    base: "vs",
    inherit: true,
    rules: [
      { token: "comment", foreground: "9aa0a6", fontStyle: "italic" },
      { token: "string", foreground: "c5221f" },
      { token: "keyword", foreground: "8430ce" },
      { token: "type", foreground: "1a73e8" },
      { token: "number", foreground: "b06000" },
      { token: "function", foreground: "0b8043" },
      { token: "identifier", foreground: "1f1f23" },
      { token: "delimiter", foreground: "5f6368" },
    ],
    colors: {
      "editor.background": "#ffffff",
      "editor.foreground": "#1f1f23",
      "editorLineNumber.foreground": "#c4c8ce",
      "editorLineNumber.activeForeground": "#9aa0a6",
      "editor.lineHighlightBackground": "#f7f8fa",
      "editor.lineHighlightBorder": "#00000000",
      "editorGutter.background": "#ffffff",
      "editor.selectionBackground": "#cfe1fb",
      "editor.inactiveSelectionBackground": "#e8f0fe",
      "editorIndentGuide.background1": "#eef0f2",
      "editorIndentGuide.activeBackground1": "#dde0e4",
      "editorCursor.foreground": "#1a73e8",
      "diffEditor.insertedTextBackground": "#34a85322",
      "diffEditor.removedTextBackground": "#ea433526",
      "diffEditor.insertedLineBackground": "#e7f5ec",
      "diffEditor.removedLineBackground": "#fdecea",
      "diffEditorGutter.insertedLineBackground": "#d6efd9",
      "diffEditorGutter.removedLineBackground": "#fbe0de",
      "diffEditor.diagonalFill": "#eef0f2",
      "diffEditorOverview.insertedForeground": "#34a85388",
      "diffEditorOverview.removedForeground": "#ea433588",
      "scrollbarSlider.background": "#0000001a",
      "scrollbarSlider.hoverBackground": "#00000026",
      "scrollbarSlider.activeBackground": "#00000033",
      "editorOverviewRuler.border": "#00000000",
      "editorWidget.background": "#ffffff",
      "editorWidget.border": "#dde0e4",
      "editorWidget.foreground": "#1f1f23",
    },
  });
}

const GL_DIFF_OPTS = {
  theme: "gaslite",
  automaticLayout: true,
  originalEditable: true, // left pane is editable
  readOnly: true, // right (optimized) pane is read-only
  renderSideBySide: true,
  enableSplitViewResizing: true,
  fontFamily: "'JetBrains Mono', ui-monospace, monospace",
  fontSize: 13,
  lineHeight: 21,
  letterSpacing: 0.1,
  minimap: { enabled: false },
  glyphMargin: false,
  folding: false,
  lineDecorationsWidth: 12,
  lineNumbersMinChars: 3,
  renderOverviewRuler: false,
  overviewRulerLanes: 0,
  scrollBeyondLastLine: false,
  smoothScrolling: true,
  cursorBlinking: "smooth",
  renderLineHighlight: "line",
  guides: { indentation: false },
  diffWordWrap: "off",
  padding: { top: 12, bottom: 18 },
  scrollbar: { vertical: "auto", verticalScrollbarSize: 10, horizontalScrollbarSize: 10, useShadows: false },
  hideUnchangedRegions: { enabled: false, contextLineCount: 3, minimumLineCount: 4, revealLineCount: 12 },
  fontLigatures: false,
} as const;

interface WhyState {
  reason: ReasonKey;
  top: number;
}

export const MonacoDiff = forwardRef<DiffHandle, { phase: Phase; optimizedSrc?: string }>(function MonacoDiff(
  { phase, optimizedSrc },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const refs = useRef<any>({});
  const [ready, setReady] = useState(false);
  const [why, setWhy] = useState<WhyState | null>(null);
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  useImperativeHandle(ref, () => ({
    getOriginal: () => (refs.current.original ? refs.current.original.getValue() : ORIGINAL_SRC),
  }));

  /* mount the diff editor once */
  useEffect(() => {
    let dead = false;
    loadMonaco()
      .then((monaco) => {
        if (dead || !hostRef.current) return;
        glSetupMonaco(monaco);
        const original = monaco.editor.createModel(ORIGINAL_SRC, "sol");
        const modified = monaco.editor.createModel(ORIGINAL_SRC, "sol");
        const diff = monaco.editor.createDiffEditor(hostRef.current, GL_DIFF_OPTS);
        diff.setModel({ original, modified });
        const me0 = diff.getModifiedEditor();
        const oe0 = diff.getOriginalEditor();
        // container can measure 0 / font not ready on first paint — force re-render
        const kick = () => {
          diff.layout();
          oe0.layout();
          me0.layout();
          oe0.render(true);
          me0.render(true);
        };
        requestAnimationFrame(kick);
        [60, 200, 500, 1000].forEach((t) => setTimeout(kick, t));
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(kick);
        const me = diff.getModifiedEditor();
        const oe = diff.getOriginalEditor();
        refs.current = { monaco, diff, original, modified, me, oe, whyByLine: {}, deco: null, flash: null, flashT: 0 };

        // while idle, keep the optimized pane mirroring edits to the original
        oe.onDidChangeModelContent(() => {
          if (phaseRef.current !== "done") modified.setValue(original.getValue());
        });

        // hover-to-explain on the optimized pane
        const updateHover = (pos: any) => {
          if (phaseRef.current !== "done" || !pos) {
            setWhy(null);
            return;
          }
          const reason = refs.current.whyByLine[pos.lineNumber];
          if (!reason) {
            setWhy(null);
            return;
          }
          const li = me.getLayoutInfo();
          const top = me.getTopForLineNumber(pos.lineNumber) - me.getScrollTop();
          setWhy({ reason, top: Math.max(6, Math.min(top, li.height - 172)) });
        };
        me.onMouseMove((e: any) => updateHover(e.target && e.target.position));
        me.onMouseLeave(() => setWhy(null));
        me.onDidScrollChange(() => setWhy(null));

        setReady(true);
      })
      .catch((err) => console.error("Monaco failed to load:", err));

    return () => {
      dead = true;
      const r = refs.current;
      if (r.flashT) clearTimeout(r.flashT);
      if (r.diff) r.diff.dispose();
      if (r.original) r.original.dispose();
      if (r.modified) r.modified.dispose();
    };
  }, []);

  /* react to phase: reveal the optimized source + annotations on "done" */
  useEffect(() => {
    if (!ready) return;
    const r = refs.current;
    const { monaco, modified, me, diff } = r;
    setWhy(null);

    // before Optimize, freeze the editor — no scroll, no edits, pinned to top
    const locked = phase !== "done";
    const lockScroll = {
      vertical: locked ? "hidden" : "auto",
      horizontal: locked ? "hidden" : "auto",
      handleMouseWheel: !locked,
      alwaysConsumeMouseWheel: false,
      verticalScrollbarSize: 10,
      horizontalScrollbarSize: 10,
      useShadows: false,
    };
    r.oe.updateOptions({ readOnly: locked, domReadOnly: locked, scrollBeyondLastLine: false, scrollbar: lockScroll });
    me.updateOptions({ scrollBeyondLastLine: false, scrollbar: lockScroll });
    if (locked) {
      r.oe.setScrollTop(0);
      r.oe.setScrollLeft(0);
      me.setScrollTop(0);
      me.setScrollLeft(0);
    }
    if (r.deco) {
      r.deco.clear();
      r.deco = null;
    }
    if (r.flash) {
      r.flash.clear();
      r.flash = null;
    }
    if (r.flashT) {
      clearTimeout(r.flashT);
      r.flashT = 0;
    }

    const repaint = () => {
      diff.layout();
      me.render(true);
      r.oe.render(true);
    };

    if (phase === "done") {
      // Optimized code comes from the live Gaslite server; fall back to mirroring
      // the original if it's missing (shouldn't happen on a successful run).
      modified.setValue(optimizedSrc ?? r.original.getValue());
      const whyByLine: Record<number, ReasonKey> = {};
      const decos: any[] = [];
      const flashes: any[] = [];
      OPTIMIZATIONS.forEach((o) => {
        const m = modified.findMatches(o.find, false, false, true, null, false);
        if (!m.length) return;
        const line = m[0].range.startLineNumber;
        whyByLine[line] = o.reason;
        decos.push({
          range: new monaco.Range(line, 1, line, 1),
          options: { isWholeLine: true, linesDecorationsClassName: "gl-why-bar", className: "gl-why-line" },
        });
        flashes.push({ range: new monaco.Range(line, 1, line, 1), options: { isWholeLine: true, className: "gl-flash" } });
      });
      r.whyByLine = whyByLine;
      r.deco = me.createDecorationsCollection(decos);
      r.flash = me.createDecorationsCollection(flashes);
      r.flashT = setTimeout(() => {
        if (r.flash) {
          r.flash.clear();
          r.flash = null;
        }
      }, 1200);
    } else {
      r.whyByLine = {};
      modified.setValue(r.original.getValue());
    }
    requestAnimationFrame(repaint);
    [60, 220, 500].forEach((t) => setTimeout(repaint, t));
  }, [phase, ready, optimizedSrc]);

  const w = why && REASONS[why.reason];
  return (
    <div className="mc-wrap">
      <div className="mc-host" ref={hostRef} />
      {!ready && <div className="mc-loading">Loading editor…</div>}
      {w && why && (
        <div className="why-pop" style={{ top: why.top, right: 22, left: "auto", pointerEvents: "none" }}>
          <div className="why-tag">{w.tag}</div>
          <div className="why-title">{w.title}</div>
          <div className="why-body">{w.body}</div>
          <div className="why-gas">
            <span>Saves</span>
            <b>{w.gas}</b>
          </div>
        </div>
      )}
    </div>
  );
});
